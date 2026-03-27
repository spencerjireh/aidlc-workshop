const STATUS_COLORS = {
  draft: 'badge-gray',
  active: 'badge-green',
  paused: 'badge-yellow',
  completed: 'badge-blue',
};

function CampaignList({ campaigns, onSelect }) {
  if (!campaigns || campaigns.length === 0) {
    return <p>No campaigns found.</p>;
  }

  return (
    <div className="card">
      <h3>Campaigns</h3>
      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Status</th>
              <th>Segments</th>
              <th>Reach</th>
            </tr>
          </thead>
          <tbody>
            {campaigns.map((c) => {
              const statusKey = (c.status || 'draft').toLowerCase();
              const badgeClass = STATUS_COLORS[statusKey] || 'badge-gray';
              return (
                <tr
                  key={c.campaign_id}
                  className="clickable-row"
                  onClick={() => onSelect(c.campaign_id)}
                >
                  <td>{c.name}</td>
                  <td>
                    <span className={`badge ${badgeClass}`}>{c.status}</span>
                  </td>
                  <td>
                    {c.segment_ids
                      ? c.segment_ids.length
                      : c.segments
                        ? c.segments.length
                        : 0}
                  </td>
                  <td>{c.reach != null ? c.reach : 'N/A'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default CampaignList;
