import threading
import time
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.logger import get_logger

logger = get_logger(__name__)


class BatchManager:
    """
    배치 처리 매니저 - GPU 활용성 향상을 위한 배치 큐잉
    - 일정 배치 크기에 도달하거나
    - 일정 시간이 경과하면 배치 처리 시작
    """
    
    def __init__(
        self, 
        max_batch_size: int = 8,
        max_wait_time: float = 2.0,  # seconds
        model_name: str = None
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.model_name = model_name
        
        # 배치별 큐: {model_name: [(job_id, data, config, metadata, timestamp)]}
        self.batch_queues: Dict[str, List[tuple]] = defaultdict(list)
        
        # 타이머: {model_name: timer_thread}
        self.timers: Dict[str, threading.Timer] = {}
        
        # 락
        self.lock = threading.Lock()
        
        logger.info(f"BatchManager initialized: max_batch_size={max_batch_size}, max_wait_time={max_wait_time}s")
    
    def add_to_batch(
        self, 
        model_name: str, 
        job_id: str, 
        data: Any, 
        config: Dict, 
        metadata: Dict,
        batch_callback: callable
    ) -> bool:
        """
        배치 큐에 작업 추가
        Returns: True if batch is ready, False if waiting
        """
        with self.lock:
            timestamp = time.time()
            
            # 큐에 작업 추가
            self.batch_queues[model_name].append((job_id, data, config, metadata, timestamp, batch_callback))
            
            queue_size = len(self.batch_queues[model_name])
            logger.info(f"[Batch] Added job {job_id} to {model_name} batch. Queue size: {queue_size}/{self.max_batch_size}")
            
            # 배치 크기 도달 시 즉시 처리
            if queue_size >= self.max_batch_size:
                logger.info(f"[Batch] Batch size reached for {model_name}. Processing immediately.")
                self._cancel_timer(model_name)
                self._process_batch(model_name)
                return True
            
            # 첫 번째 작업이면 타이머 시작
            if queue_size == 1:
                self._start_timer(model_name)
            
            return False
    
    def _start_timer(self, model_name: str):
        """타이머 시작 - max_wait_time 후 배치 처리"""
        def timeout_callback():
            with self.lock:
                if model_name in self.batch_queues and len(self.batch_queues[model_name]) > 0:
                    logger.info(f"[Batch] Timeout reached for {model_name}. Processing batch.")
                    self._process_batch(model_name)
        
        timer = threading.Timer(self.max_wait_time, timeout_callback)
        timer.daemon = True
        timer.start()
        self.timers[model_name] = timer
        
        logger.debug(f"[Batch] Timer started for {model_name}: {self.max_wait_time}s")
    
    def _cancel_timer(self, model_name: str):
        """타이머 취소"""
        if model_name in self.timers:
            self.timers[model_name].cancel()
            del self.timers[model_name]
            logger.debug(f"[Batch] Timer cancelled for {model_name}")
    
    def _process_batch(self, model_name: str):
        """배치 처리 시작"""
        if model_name not in self.batch_queues or len(self.batch_queues[model_name]) == 0:
            return
        
        # 현재 배치 가져오기
        batch = self.batch_queues[model_name]
        self.batch_queues[model_name] = []
        
        logger.info(f"[Batch] Processing batch for {model_name}: {len(batch)} jobs")
        
        # 각 작업의 콜백 실행
        for job_id, data, config, metadata, timestamp, batch_callback in batch:
            wait_time = time.time() - timestamp
            logger.info(f"[Batch] Job {job_id} waited {wait_time:.2f}s in queue")
            
            # 배치 처리 콜백 실행 (비동기)
            threading.Thread(
                target=batch_callback,
                args=(job_id, model_name, data, config, metadata),
                daemon=True
            ).start()
    
    def get_queue_status(self) -> Dict[str, int]:
        """현재 큐 상태 조회"""
        with self.lock:
            return {model: len(queue) for model, queue in self.batch_queues.items()}
    
    def clear_queue(self, model_name: str = None):
        """큐 초기화"""
        with self.lock:
            if model_name:
                self.batch_queues[model_name] = []
                self._cancel_timer(model_name)
            else:
                self.batch_queues.clear()
                for timer in self.timers.values():
                    timer.cancel()
                self.timers.clear()


batch_manager = BatchManager(
    max_batch_size=8,
    max_wait_time=2.0
)
