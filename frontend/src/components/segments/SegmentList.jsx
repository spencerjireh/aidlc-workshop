function SegmentList({ segments, onSelect }) {
  if (!segments || segments.length === 0) {
    return <p>No segments found.</p>;
  }

  return (
    <div className="card">
      <h3>Customer Segments</h3>
      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Size</th>
              <th>Avg Transaction</th>
              <th>Top Categories</th>
            </tr>
          </thead>
          <tbody>
            {segments.map((seg) => (
              <tr
                key={seg.segment_id}
                className="clickable-row"
                onClick={() => onSelect(seg.segment_id)}
              >
                <td>{seg.name}</td>
                <td>{seg.size}</td>
                <td>
                  {seg.avg_transaction_amount != null
                    ? `$${Number(seg.avg_transaction_amount).toFixed(2)}`
                    : 'N/A'}
                </td>
                <td>
                  {seg.top_categories
                    ? seg.top_categories.join(', ')
                    : 'N/A'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default SegmentList;
