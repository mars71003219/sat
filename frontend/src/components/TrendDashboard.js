import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { format, subHours } from 'date-fns';
import './TrendDashboard.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api/v1';

const METRICS = [
  { key: 'satellite_temperature', label: 'Temperature', unit: '°C', color: '#4a9eff' },
  { key: 'satellite_altitude', label: 'Altitude', unit: 'km', color: '#a78bfa' },
  { key: 'satellite_velocity', label: 'Velocity', unit: 'km/s', color: '#fbbf24' },
  { key: 'satellite_battery_voltage', label: 'Battery', unit: 'V', color: '#4ade80' },
  { key: 'satellite_solar_power', label: 'Solar Power', unit: 'W', color: '#fb923c' }
];

// 메트릭별 최신 값을 저장하는 상태 추가 예정

const TIME_RANGES = [
  { label: '1m', hours: 1/60 },      // 1분
  { label: '5m', hours: 5/60 },      // 5분
  { label: '15m', hours: 15/60 },    // 15분
  { label: '1h', hours: 1 },
  { label: '6h', hours: 6 },
  { label: '1d', hours: 24 },
];

function TrendDashboard() {
  const [selectedMetric, setSelectedMetric] = useState(METRICS[0]);
  const [selectedTimeRange, setSelectedTimeRange] = useState(TIME_RANGES[1]); // 5m (실시간 느낌)
  const [rawData, setRawData] = useState([]);
  const [predictionData, setPredictionData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const [satellites, setSatellites] = useState([]);
  const [selectedSatellite, setSelectedSatellite] = useState('SAT-001');
  const [latestValues, setLatestValues] = useState({});

  // 위성 목록 조회
  useEffect(() => {
    fetchSatellites();
  }, []);

  const fetchSatellites = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/trends/satellites`);
      const data = await response.json();
      if (data.satellites && data.satellites.length > 0) {
        setSatellites(data.satellites);
        setSelectedSatellite(data.satellites[0]);
      }
    } catch (err) {
      console.error('Failed to fetch satellites:', err);
    }
  };

  // 데이터 조회
  const fetchTrendData = useCallback(async () => {
    setLoading(true);
    setError(null);

    const endTime = new Date();
    const startTime = subHours(endTime, selectedTimeRange.hours);

    try {
      // 원본 데이터 조회
      const rawParams = new URLSearchParams({
        metric: selectedMetric.key,
        start_time: startTime.toISOString(),
        end_time: endTime.toISOString(),
        satellite_id: selectedSatellite
      });

      const rawResponse = await fetch(`${API_BASE_URL}/trends/raw?${rawParams}`);
      if (!rawResponse.ok) throw new Error('Failed to fetch raw data');
      const rawResult = await rawResponse.json();

      // 데이터 포맷팅
      const formattedRawData = rawResult.data_points.map(point => ({
        timestamp: new Date(point.timestamp).getTime(),
        value: point.value
      }));

      setRawData(formattedRawData);
      setStats(rawResult.summary);

      // 예측 데이터 조회 (선택사항)
      // TODO: 모델 이름을 선택할 수 있도록 UI 추가
      // 임시로 비활성화 - API 에러 수정 후 재활성화
      /*
      try {
        const predParams = new URLSearchParams({
          model_name: 'vae_timeseries',
          start_time: startTime.toISOString(),
          end_time: endTime.toISOString(),
          satellite_id: selectedSatellite
        });

        const predResponse = await fetch(`${API_BASE_URL}/trends/prediction?${predParams}`);
        if (predResponse.ok) {
          const predResult = await predResponse.json();
          const formattedPredData = predResult.data_points.map(point => ({
            timestamp: new Date(point.timestamp).getTime(),
            value: point.value
          }));
          setPredictionData(formattedPredData);
        }
      } catch (predErr) {
        console.warn('Prediction data not available:', predErr);
        setPredictionData([]);
      }
      */
      setPredictionData([]); // 예측 데이터 임시 비활성화

    } catch (err) {
      setError(err.message);
      console.error('Error fetching trend data:', err);
    } finally {
      setLoading(false);
    }
  }, [selectedMetric, selectedTimeRange, selectedSatellite]);

  // 모든 메트릭의 최신 값을 가져오는 함수
  const fetchLatestValues = useCallback(async () => {
    try {
      const promises = METRICS.map(async (metric) => {
        const response = await fetch(
          `/victoriametrics/api/v1/query?query=${metric.key}{satellite_id="${selectedSatellite}"}`
        );
        const data = await response.json();

        if (data.status === 'success' && data.data.result.length > 0) {
          const value = parseFloat(data.data.result[0].value[1]);
          return { key: metric.key, value };
        }
        return { key: metric.key, value: null };
      });

      const results = await Promise.all(promises);
      const valuesMap = {};
      results.forEach(r => {
        valuesMap[r.key] = r.value;
      });
      setLatestValues(valuesMap);
    } catch (err) {
      console.error('Failed to fetch latest values:', err);
    }
  }, [selectedSatellite]);

  useEffect(() => {
    fetchTrendData();
    fetchLatestValues();

    // 자동 새로고침: 트렌드 차트와 최신 값 모두 1초마다 업데이트
    const trendInterval = setInterval(fetchTrendData, 1000);
    const valuesInterval = setInterval(fetchLatestValues, 1000);

    return () => {
      clearInterval(trendInterval);
      clearInterval(valuesInterval);
    };
  }, [fetchTrendData, fetchLatestValues]);

  // 차트 데이터 병합
  const mergedChartData = React.useMemo(() => {
    const dataMap = new Map();

    rawData.forEach(point => {
      dataMap.set(point.timestamp, { timestamp: point.timestamp, raw: point.value });
    });

    predictionData.forEach(point => {
      const existing = dataMap.get(point.timestamp) || { timestamp: point.timestamp };
      existing.prediction = point.value;
      dataMap.set(point.timestamp, existing);
    });

    return Array.from(dataMap.values()).sort((a, b) => a.timestamp - b.timestamp);
  }, [rawData, predictionData]);

  const formatXAxis = (timestamp) => {
    // 시간 범위에 따라 포맷 변경
    if (selectedTimeRange.hours <= 0.25) { // 15분 이하: 초 단위 표시
      return format(new Date(timestamp), 'HH:mm:ss');
    } else if (selectedTimeRange.hours <= 1) { // 1시간 이하: 분 단위 표시
      return format(new Date(timestamp), 'HH:mm');
    } else { // 그 이상: 시간 단위 표시
      return format(new Date(timestamp), 'HH:mm');
    }
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="tooltip-time">{format(new Date(label), 'yyyy-MM-dd HH:mm:ss')}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {entry.value.toFixed(2)} {selectedMetric.unit}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="trend-dashboard">
      {/* 헤더 */}
      <header className="dashboard-header">
        <div className="header-left">
          <h1>🛰️ Satellite Monitor</h1>
          {selectedSatellite && (
            <span className="satellite-badge">{selectedSatellite}</span>
          )}
        </div>

        <div className="header-controls">
          {/* 위성 선택 */}
          {satellites.length > 0 && (
            <select
              className="satellite-selector"
              value={selectedSatellite}
              onChange={(e) => setSelectedSatellite(e.target.value)}
            >
              {satellites.map(sat => (
                <option key={sat} value={sat}>{sat}</option>
              ))}
            </select>
          )}

          {/* 시간 범위 선택 */}
          {TIME_RANGES.map(range => (
            <button
              key={range.label}
              className={`time-range-btn ${selectedTimeRange.label === range.label ? 'active' : ''}`}
              onClick={() => setSelectedTimeRange(range)}
            >
              {range.label}
            </button>
          ))}

          <button className="refresh-btn" onClick={fetchTrendData} disabled={loading}>
            {loading ? '⟳' : '↻'}
          </button>
        </div>
      </header>

      {/* 메트릭 카드 */}
      <div className="metrics-grid">
        {METRICS.map(metric => {
          const latestValue = latestValues[metric.key];
          const isSelected = selectedMetric.key === metric.key;

          return (
            <div
              key={metric.key}
              className={`metric-card ${isSelected ? 'selected' : ''}`}
              onClick={() => setSelectedMetric(metric)}
            >
              <div className="metric-label">{metric.label}</div>
              <div className="metric-stats">
                <div className="stat-value" style={{ color: metric.color }}>
                  {latestValue !== null && latestValue !== undefined
                    ? latestValue.toFixed(2)
                    : 'N/A'}{' '}
                  {metric.unit}
                </div>
                {isSelected && stats && (
                  <div className="stat-range">
                    Min: {stats.min?.toFixed(2) ?? 'N/A'} | Max: {stats.max?.toFixed(2) ?? 'N/A'}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* 에러 메시지 */}
      {error && (
        <div className="error-message">
          ⚠️ Error: {error}
        </div>
      )}

      {/* 메인 차트 */}
      <div className="chart-container">
        <div className="chart-header">
          <h2>{selectedMetric.label} Trend</h2>
          <div className="chart-legend">
            <span className="legend-item">
              <span className="legend-line raw"></span>
              Raw Data
            </span>
            {predictionData.length > 0 && (
              <span className="legend-item">
                <span className="legend-line prediction"></span>
                Prediction
              </span>
            )}
          </div>
        </div>

        {loading && mergedChartData.length === 0 ? (
          <div className="loading-state">Loading data...</div>
        ) : mergedChartData.length === 0 ? (
          <div className="empty-state">No data available</div>
        ) : (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={mergedChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#3a3a3f" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={formatXAxis}
                stroke="#a0a0a0"
              />
              <YAxis
                stroke="#a0a0a0"
                label={{ value: selectedMetric.unit, angle: -90, position: 'insideLeft', fill: '#a0a0a0' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Line
                type="monotone"
                dataKey="raw"
                stroke={selectedMetric.color}
                strokeWidth={2}
                dot={false}
                name="Raw Data"
                isAnimationActive={false}
              />
              {predictionData.length > 0 && (
                <Line
                  type="monotone"
                  dataKey="prediction"
                  stroke="#4ade80"
                  strokeWidth={2}
                  strokeDasharray="8 4"
                  dot={false}
                  name="Prediction"
                  isAnimationActive={false}
                  opacity={0.8}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* 통계 패널 */}
      {stats && (
        <div className="stats-panel">
          <h3>Statistics</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <div className="stat-label">Data Points</div>
              <div className="stat-value">{stats.count}</div>
            </div>
            <div className="stat-item">
              <div className="stat-label">Mean</div>
              <div className="stat-value">{stats.mean?.toFixed(2) ?? 'N/A'} {selectedMetric.unit}</div>
            </div>
            <div className="stat-item">
              <div className="stat-label">Std Dev</div>
              <div className="stat-value">{stats.std?.toFixed(2) ?? 'N/A'}</div>
            </div>
            <div className="stat-item">
              <div className="stat-label">Range</div>
              <div className="stat-value">
                {stats.min?.toFixed(2) ?? 'N/A'} ~ {stats.max?.toFixed(2) ?? 'N/A'}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default TrendDashboard;
