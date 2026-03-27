import { useEffect, useState } from 'react';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { getSegmentDistribution } from '../../api/client';

ChartJS.register(ArcElement, Tooltip, Legend);

const COLORS = [
  '#4e79a7',
  '#f28e2b',
  '#e15759',
  '#76b7b2',
  '#59a14f',
  '#edc948',
  '#b07aa1',
  '#ff9da7',
  '#9c755f',
  '#bab0ac',
];

function SegmentDistribution() {
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getSegmentDistribution()
      .then((data) => {
        const labels = data.segments
          ? data.segments.map((s) => s.name || s.segment_id)
          : [];
        const values = data.segments
          ? data.segments.map((s) => s.size || s.count || 0)
          : [];

        setChartData({
          labels,
          datasets: [
            {
              data: values,
              backgroundColor: COLORS.slice(0, labels.length),
              borderWidth: 1,
            },
          ],
        });
      })
      .catch((err) => {
        setError(err.response?.data?.detail || err.message);
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  if (loading) return <p>Loading distribution...</p>;
  if (error) return <p className="error-text">Error: {error}</p>;
  if (!chartData) return null;

  return (
    <div className="card">
      <h3>Customer Distribution</h3>
      <div className="chart-container">
        <Pie
          data={chartData}
          options={{
            responsive: true,
            plugins: {
              legend: { position: 'bottom' },
            },
          }}
        />
      </div>
    </div>
  );
}

export default SegmentDistribution;
