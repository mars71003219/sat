#!/bin/bash

# .env 파일이 존재하지 않을 경우에만 새로 생성
if [ ! -f .env ]; then
  CLUSTER_ID=$(docker run --rm confluentinc/cp-kafka:latest kafka-storage random-uuid)
  echo "KAFKA_CLUSTER_ID=$CLUSTER_ID" > .env
  echo "Kafka 클러스터 ID를 생성하여 .env 파일에 저장했습니다: $CLUSTER_ID"
else
  echo "이미 .env 파일이 존재합니다."
  cat .env
fi

echo ""
echo "현재 Kafka Cluster ID:"
cat .env