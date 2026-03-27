import { useEffect, useState } from 'react';
import {
  getCampaign,
  getCampaignReach,
  getCampaignPerformance,
  activateCampaign,
} from '../../api/client';

function CampaignDetail({ campaignId, onClose, onUpdated }) {
  const [campaign, setCampaign] = useState(null);
  const [reach, setReach] = useState(null);
  const [performance, setPerformance] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activating, setActivating] = useState(false);

  useEffect(() => {
    if (!campaignId) return;
    setLoading(true);
    setError(null);

    Promise.all([
      getCampaign(campaignId),
      getCampaignReach(campaignId).catch(() => null),
      getCampaignPerformance(campaignId).catch(() => null),
    ])
      .then(([campaignData, reachData, perfData]) => {
        setCampaign(campaignData);
        setReach(reachData);
        setPerformance(perfData);
      })
      .catch((err) => {
        setError(err.response?.data?.detail || err.message);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [campaignId]);

  async function handleActivate() {
    setActivating(true);
    try {
      const result = await activateCampaign(campaignId);
      setCampaign(result);
      if (onUpdated) onUpdated();
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setActivating(false);
    }
  }

  if (loading) return <p>Loading campaign details...</p>;
  if (error) return <p className="error-text">Error: {error}</p>;
  if (!campaign) return null;

  const statusKey = (campaign.status || 'draft').toLowerCase();

  return (
    <div className="card campaign-detail">
      <div className="detail-header">
        <h3>{campaign.name}</h3>
        <button className="btn btn-secondary" onClick={onClose}>
          Close
        </button>
      </div>

      <div className="detail-meta">
        <span>
          <strong>Status:</strong> {campaign.status}
        </span>
        {reach && (
          <span>
            <strong>Total Reach:</strong> {reach.total_reach || reach.reach || 'N/A'}
          </span>
        )}
      </div>

      {campaign.segment_ids && (
        <div className="detail-section">
          <h4>Segments</h4>
          <ul>
            {campaign.segment_ids.map((sid) => (
              <li key={sid}>{sid}</li>
            ))}
          </ul>
        </div>
      )}

      {statusKey === 'draft' && (
        <button
          className="btn btn-primary"
          onClick={handleActivate}
          disabled={activating}
        >
          {activating ? 'Activating...' : 'Activate Campaign'}
        </button>
      )}

      {performance && performance.ads && (
        <div className="detail-section">
          <h4>Performance Metrics</h4>
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ad ID</th>
                  <th>Impressions</th>
                  <th>Clicks</th>
                  <th>CTR</th>
                  <th>Conversions</th>
                </tr>
              </thead>
              <tbody>
                {performance.ads.map((ad) => (
                  <tr key={ad.ad_id || ad.id}>
                    <td>{ad.ad_id || ad.id}</td>
                    <td>{ad.impressions ?? 'N/A'}</td>
                    <td>{ad.clicks ?? 'N/A'}</td>
                    <td>
                      {ad.ctr != null
                        ? `${(ad.ctr * 100).toFixed(2)}%`
                        : 'N/A'}
                    </td>
                    <td>{ad.conversions ?? 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default CampaignDetail;
