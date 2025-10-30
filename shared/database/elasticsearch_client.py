from elasticsearch import Elasticsearch
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ElasticsearchClient:
    """Elasticsearch 클라이언트 - 로그 및 검색"""
    
    def __init__(self):
        self.es = None
        self._initialize()
    
    def _initialize(self):
        """Elasticsearch 연결 초기화"""
        try:
            self.es = Elasticsearch(
                [settings.ELASTICSEARCH_URL],
                verify_certs=False
            )
            if self.es.ping():
                logger.info("Elasticsearch connection established")
                self._create_indices()
            else:
                logger.warning("Elasticsearch ping failed")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            self.es = None
    
    def _create_indices(self):
        """인덱스 생성"""
        indices = {
            'inference_logs': {
                'mappings': {
                    'properties': {
                        'job_id': {'type': 'keyword'},
                        'model_name': {'type': 'keyword'},
                        'status': {'type': 'keyword'},
                        'predictions': {'type': 'float'},
                        'inference_time': {'type': 'float'},
                        'created_at': {'type': 'date'},
                        'completed_at': {'type': 'date'},
                        'metadata': {'type': 'object', 'enabled': False}
                    }
                }
            },
            'activity_logs': {
                'mappings': {
                    'properties': {
                        'user_id': {'type': 'keyword'},
                        'action': {'type': 'keyword'},
                        'resource_type': {'type': 'keyword'},
                        'resource_id': {'type': 'keyword'},
                        'timestamp': {'type': 'date'},
                        'ip_address': {'type': 'ip'},
                        'details': {'type': 'object', 'enabled': False}
                    }
                }
            },
            'sensor_data': {
                'mappings': {
                    'properties': {
                        'sensor_id': {'type': 'keyword'},
                        'sensor_type': {'type': 'keyword'},
                        'location': {'type': 'keyword'},
                        'value': {'type': 'float'},
                        'unit': {'type': 'keyword'},
                        'timestamp': {'type': 'date'}
                    }
                }
            }
        }
        
        for index_name, index_body in indices.items():
            try:
                if not self.es.indices.exists(index=index_name):
                    self.es.indices.create(index=index_name, body=index_body)
                    logger.info(f"Created index: {index_name}")
            except Exception as e:
                logger.error(f"Failed to create index {index_name}: {e}")
    
    def index_inference_result(self, result: Dict[str, Any]):
        """추론 결과 인덱싱"""
        if not self.es:
            return
        
        try:
            doc = {
                'job_id': result['job_id'],
                'model_name': result['model_name'],
                'status': result['status'],
                'predictions': result.get('predictions', []),
                'inference_time': result.get('metrics', {}).get('inference_time'),
                'created_at': result['created_at'],
                'completed_at': result.get('completed_at'),
                'metadata': result.get('metadata', {})
            }
            
            self.es.index(index='inference_logs', document=doc, id=result['job_id'])
            logger.debug(f"Indexed inference result: {result['job_id']}")
        except Exception as e:
            logger.error(f"Failed to index inference result: {e}")
    
    def index_activity_log(self, log: Dict[str, Any]):
        """활동 로그 인덱싱"""
        if not self.es:
            return
        
        try:
            self.es.index(index='activity_logs', document=log)
            logger.debug(f"Indexed activity log: {log.get('action')}")
        except Exception as e:
            logger.error(f"Failed to index activity log: {e}")
    
    def index_sensor_data(self, data: Dict[str, Any]):
        """센서 데이터 인덱싱"""
        if not self.es:
            return
        
        try:
            self.es.index(index='sensor_data', document=data)
        except Exception as e:
            logger.error(f"Failed to index sensor data: {e}")
    
    def search_inference_logs(
        self, 
        model_name: Optional[str] = None,
        status: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """추론 로그 검색"""
        if not self.es:
            return []
        
        query = {'bool': {'must': []}}
        
        if model_name:
            query['bool']['must'].append({'term': {'model_name': model_name}})
        if status:
            query['bool']['must'].append({'term': {'status': status}})
        if from_date or to_date:
            range_query = {'created_at': {}}
            if from_date:
                range_query['created_at']['gte'] = from_date
            if to_date:
                range_query['created_at']['lte'] = to_date
            query['bool']['must'].append({'range': range_query})
        
        if not query['bool']['must']:
            query = {'match_all': {}}
        
        try:
            result = self.es.search(
                index='inference_logs',
                query=query,
                size=limit,
                sort=[{'created_at': {'order': 'desc'}}]
            )
            return [hit['_source'] for hit in result['hits']['hits']]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def aggregate_by_model(self) -> Dict[str, Any]:
        """모델별 통계 집계"""
        if not self.es:
            return {}
        
        try:
            result = self.es.search(
                index='inference_logs',
                size=0,
                aggs={
                    'models': {
                        'terms': {'field': 'model_name', 'size': 50},
                        'aggs': {
                            'avg_inference_time': {
                                'avg': {'field': 'inference_time'}
                            },
                            'status_breakdown': {
                                'terms': {'field': 'status'}
                            }
                        }
                    }
                }
            )
            return result['aggregations']
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return {}
    
    def search_sensor_data(
        self,
        sensor_id: Optional[str] = None,
        sensor_type: Optional[str] = None,
        location: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """센서 데이터 검색"""
        if not self.es:
            return []
        
        query = {'bool': {'must': []}}
        
        if sensor_id:
            query['bool']['must'].append({'term': {'sensor_id': sensor_id}})
        if sensor_type:
            query['bool']['must'].append({'term': {'sensor_type': sensor_type}})
        if location:
            query['bool']['must'].append({'term': {'location': location}})
        if from_time or to_time:
            range_query = {'timestamp': {}}
            if from_time:
                range_query['timestamp']['gte'] = from_time
            if to_time:
                range_query['timestamp']['lte'] = to_time
            query['bool']['must'].append({'range': range_query})
        
        if not query['bool']['must']:
            query = {'match_all': {}}
        
        try:
            result = self.es.search(
                index='sensor_data',
                query=query,
                size=limit,
                sort=[{'timestamp': {'order': 'desc'}}]
            )
            return [hit['_source'] for hit in result['hits']['hits']]
        except Exception as e:
            logger.error(f"Sensor data search failed: {e}")
            return []


elasticsearch_client = ElasticsearchClient()
