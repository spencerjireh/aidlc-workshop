import { NavLink } from 'react-router-dom';

function Sidebar() {
  return (
    <nav className="sidebar">
      <div className="sidebar-header">
        <h2>AIDLC Workshop</h2>
      </div>
      <ul className="sidebar-nav">
        <li>
          <NavLink
            to="/segments"
            className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}
          >
            Segments
          </NavLink>
        </li>
        <li>
          <NavLink
            to="/campaigns"
            className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}
          >
            Campaigns
          </NavLink>
        </li>
        <li>
          <NavLink
            to="/chatbot"
            className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}
          >
            Chatbot
          </NavLink>
        </li>
      </ul>
    </nav>
  );
}

export default Sidebar;
