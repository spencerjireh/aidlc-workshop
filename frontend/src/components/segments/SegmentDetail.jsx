import { useEffect, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from 'chart.js';
import { getSegment } from '../../api/client';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

function SegmentDetail({ segmentId, onClose }) {
  const [segment, setSegment] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!segmentId) return;
    setLoading(true);
    setError(null);

    getSegment(segmentId)
      .then((data) => {
        setSegment(data);
      })
      .catch((err) => {
        setError(err.response?.data?.detail || err.message);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [segmentId]);

  if (loading) return <p>Loading segment details...</p>;
  if (error) return <p className="error-text">Error: {error}</p>;
  if (!segment) return null;

  const pcaData = segment.pca_contributions || segment.differentiating_factors;
  let barChartData = null;

  if (pcaData && typeof pcaData === 'object') {
    const labels = Object.keys(pcaData);
    const values = Object.values(pcaData);

    barChartData = {
      labels,
      datasets: [
        {
          label: 'Contribution',
          data: values,
          backgroundColor: '#4e79a7',
        },
      ],
    };
  }

  return (
    <div className="card segment-detail">
      <div className="detail-header">
        <h3>{segment.name}</h3>
        <button className="btn btn-secondary" onClick={onClose}>
          Close
        </button>
      </div>
      {segment.description && <p>{segment.description}</p>}
      <div className="detail-meta">
        <span>
          <strong>Size:</strong> {segment.size}
        </span>
        {segment.avg_transaction_amount != null && (
          <span>
            <strong>Avg Transaction:</strong> ${Number(segment.avg_transaction_amount).toFixed(2)}
          </span>
        )}
      </div>
      {segment.differentiating_factors && (
        <div className="detail-section">
          <h4>Differentiating Factors</h4>
          <ul>
            {Array.isArray(segment.differentiating_factors)
              ? segment.differentiating_factors.map((f, i) => <li key={i}>{f}</li>)
              : Object.entries(segment.differentiating_factors).map(([k, v]) => (
                  <li key={k}>
                    {k}: {typeof v === 'number' ? v.toFixed(4) : String(v)}
                  </li>
                ))}
          </ul>
        </div>
      )}
      {barChartData && (
        <div className="detail-section">
          <h4>PCA Component Contributions</h4>
          <div className="chart-container">
            <Bar
              data={barChartData}
              options={{
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                  y: { beginAtZero: true },
                },
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default SegmentDetail;
