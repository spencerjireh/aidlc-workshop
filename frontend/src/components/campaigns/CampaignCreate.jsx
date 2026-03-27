import { useState, useEffect } from 'react';
import { createCampaign, getSegments } from '../../api/client';

function CampaignCreate({ onCreated }) {
  const [name, setName] = useState('');
  const [selectedSegments, setSelectedSegments] = useState([]);
  const [adContentIds, setAdContentIds] = useState('');
  const [segments, setSegments] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  useEffect(() => {
    getSegments()
      .then((data) => {
        setSegments(Array.isArray(data) ? data : data.segments || []);
      })
      .catch(() => {
        // segments list not critical for form render
      });
  }, []);

  function handleSegmentToggle(segmentId) {
    setSelectedSegments((prev) =>
      prev.includes(segmentId)
        ? prev.filter((id) => id !== segmentId)
        : [...prev, segmentId]
    );
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    if (!name.trim()) {
      setError('Campaign name is required.');
      return;
    }
    if (selectedSegments.length === 0) {
      setError('Select at least one segment.');
      return;
    }

    setSubmitting(true);

    try {
      const payload = {
        name: name.trim(),
        segment_ids: selectedSegments,
      };

      const ids = adContentIds
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
      if (ids.length > 0) {
        payload.ad_content_ids = ids;
      }

      const result = await createCampaign(payload);
      setSuccess(`Campaign "${result.name || name}" created successfully.`);
      setName('');
      setSelectedSegments([]);
      setAdContentIds('');
      if (onCreated) onCreated();
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="card">
      <h3>Create Campaign</h3>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="campaign-name">Campaign Name</label>
          <input
            id="campaign-name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Enter campaign name"
          />
        </div>

        <div className="form-group">
          <label>Select Segments</label>
          <div className="checkbox-group">
            {segments.map((seg) => (
              <label key={seg.segment_id} className="checkbox-label">
                <input
                  type="checkbox"
                  checked={selectedSegments.includes(seg.segment_id)}
                  onChange={() => handleSegmentToggle(seg.segment_id)}
                />
                {seg.name} ({seg.size})
              </label>
            ))}
            {segments.length === 0 && <p>No segments available.</p>}
          </div>
        </div>

        <div className="form-group">
          <label htmlFor="ad-content-ids">Ad Content IDs (comma-separated)</label>
          <input
            id="ad-content-ids"
            type="text"
            value={adContentIds}
            onChange={(e) => setAdContentIds(e.target.value)}
            placeholder="e.g. ad_001, ad_002"
          />
        </div>

        {error && <p className="error-text">{error}</p>}
        {success && <p className="success-text">{success}</p>}

        <button type="submit" className="btn btn-primary" disabled={submitting}>
          {submitting ? 'Creating...' : 'Create Campaign'}
        </button>
      </form>
    </div>
  );
}

export default CampaignCreate;
