import { useEffect, useState, useCallback } from 'react';
import { getCampaigns } from '../api/client';
import CampaignList from '../components/campaigns/CampaignList';
import CampaignCreate from '../components/campaigns/CampaignCreate';
import CampaignDetail from '../components/campaigns/CampaignDetail';

function CampaignsPage() {
  const [campaigns, setCampaigns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedCampaignId, setSelectedCampaignId] = useState(null);

  const fetchCampaigns = useCallback(() => {
    setLoading(true);
    setError(null);
    getCampaigns()
      .then((data) => {
        setCampaigns(Array.isArray(data) ? data : data.campaigns || []);
      })
      .catch((err) => {
        setError(err.response?.data?.detail || err.message);
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    fetchCampaigns();
  }, [fetchCampaigns]);

  return (
    <div className="page">
      <h2>Campaigns</h2>
      {loading && <p>Loading...</p>}
      {error && <p className="error-text">Error: {error}</p>}

      {!loading && !error && (
        <div className="campaigns-layout">
          <div className="campaigns-main">
            <CampaignList
              campaigns={campaigns}
              onSelect={setSelectedCampaignId}
            />
          </div>
          <div className="campaigns-sidebar">
            <CampaignCreate onCreated={fetchCampaigns} />
          </div>
        </div>
      )}

      {selectedCampaignId && (
        <CampaignDetail
          campaignId={selectedCampaignId}
          onClose={() => setSelectedCampaignId(null)}
          onUpdated={fetchCampaigns}
        />
      )}
    </div>
  );
}

export default CampaignsPage;
