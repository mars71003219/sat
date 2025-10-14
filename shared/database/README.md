# Database Initialization

This directory contains SQL scripts that are automatically executed by PostgreSQL on first startup.

## How It Works

When the PostgreSQL container starts for the first time, it automatically executes all `.sql` and `.sh` files in `/docker-entrypoint-initdb.d/` in alphabetical order.

The docker-compose.yml file mounts this directory:
```yaml
volumes:
  - ./shared/database:/docker-entrypoint-initdb.d
```

## Files

### 01-schema.sql
Creates the database schema with the following tables:
- **inference_results**: AI inference job results
- **sensor_data**: IoT sensor measurements
- **timeseries_datasets**: Time series training datasets
- **model_metrics**: Model performance metrics
- **activity_logs**: User activity and system events

Also creates views:
- **prediction_history**: Historical predictions with processing times
- **model_performance_summary**: Aggregated model statistics

### 02-seed-data.sql
Populates tables with sample data:
- 15+ sensor data entries (temperature, humidity, pressure, vibration, flow rate)
- 5 timeseries datasets (linear, seasonal, random walk, cyclical, exponential patterns)
- 15+ model metrics entries for LSTM and Moving Average models
- 15+ activity log entries simulating user actions
- 4 sample inference results

## Automatic Execution

The scripts run automatically when:
1. Starting containers with `docker compose up -d` for the first time
2. After removing the postgres_data volume and restarting

## Manual Execution

To re-initialize the database:
```bash
docker compose down -v
docker compose up -d
```

The `-v` flag removes volumes, forcing PostgreSQL to reinitialize.

## Verification

Check if initialization succeeded:
```bash
docker compose exec postgres psql -U admin -d orders_db -c "SELECT COUNT(*) FROM inference_results;"
```

Expected output: 4 rows (sample inference results)
