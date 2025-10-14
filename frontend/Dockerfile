# ------------------ STAGE 1: Build ------------------
# React 앱을 빌드하기 위한 빌드 환경을 정의합니다.
FROM node:18-alpine AS build

# 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app

# package.json과 package-lock.json을 먼저 복사하여
# 의존성이 변경되지 않았다면 Docker 캐시를 활용합니다.
COPY package.json ./
COPY package-lock.json ./

# 의존성을 설치합니다.
RUN npm install

# 나머지 모든 소스 코드를 복사합니다.
COPY . ./

# 프로덕션용으로 앱을 빌드합니다.
RUN npm run build


# ------------------ STAGE 2: Serve ------------------
# 빌드된 결과물을 서빙하기 위한 최종 환경을 정의합니다.
FROM nginx:alpine

# Stage 1(build)에서 생성된 빌드 결과물을
# Nginx의 기본 웹 루트 디렉토리로 복사합니다.
COPY --from=build /app/build /usr/share/nginx/html

# Nginx가 80번 포트를 사용하도록 설정합니다.
EXPOSE 80

# 컨테이너가 시작될 때 Nginx를 실행합니다.
CMD ["nginx", "-g", "daemon off;"]