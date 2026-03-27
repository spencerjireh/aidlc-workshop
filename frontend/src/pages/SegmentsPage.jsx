import { useEffect, useState } from 'react';
import { getSegments } from '../api/client';
import SegmentList from '../components/segments/SegmentList';
import SegmentDistribution from '../components/segments/SegmentDistribution';
import SegmentDetail from '../components/segments/SegmentDetail';

function SegmentsPage() {
  const [segments, setSegments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedSegmentId, setSelectedSegmentId] = useState(null);

  useEffect(() => {
    getSegments()
      .then((data) => {
        setSegments(Array.isArray(data) ? data : data.segments || []);
      })
      .catch((err) => {
        setError(err.response?.data?.detail || err.message);
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  return (
    <div className="page">
      <h2>Segments</h2>
      {loading && <p>Loading...</p>}
      {error && <p className="error-text">Error: {error}</p>}

      {!loading && !error && (
        <div className="segments-layout">
          <div className="segments-main">
            <SegmentList segments={segments} onSelect={setSelectedSegmentId} />
          </div>
          <div className="segments-sidebar">
            <SegmentDistribution />
          </div>
        </div>
      )}

      {selectedSegmentId && (
        <SegmentDetail
          segmentId={selectedSegmentId}
          onClose={() => setSelectedSegmentId(null)}
        />
      )}
    </div>
  );
}

export default SegmentsPage;
