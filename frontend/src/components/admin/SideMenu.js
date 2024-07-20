import React from 'react'
import { Link } from 'react-router-dom'
import { useCookies } from 'react-cookie';

const SideMenu = () => {
  const [cookies] = useCookies(['adminUsername']);
  return (
    <>
    {/* ======= Sidebar ======= */}
<aside id="sidebar" className="sidebar">
  <ul className="sidebar-nav" id="sidebar-nav">
    <li className="nav-item">
      <Link className="nav-link collapsed " to="/admin/dashboard">
        <i className="bi bi-grid" />
        <span>Dashboard</span>
      </Link>
    </li>{/* End Dashboard Nav */}
    <li className="nav-item">
      <Link className="nav-link collapsed" to="/admin/profile">
        <i className="bi bi-person" />
        <span>Profile</span>
      </Link>
    </li>{/* End Profile Page Nav */}
   
    {cookies.adminUsername === 'Vishesh12' && (
            <li className="nav-item">
              <Link className="nav-link collapsed" to="/admin/register">
                <i className="bi bi-person-lock" />
                <span>Register An Admin</span>
              </Link>
            </li>
          )}
   
    <li className="nav-item">
      <Link className="nav-link collapsed" to="/admin/bank-register">
        <i className="bi bi-bank2" />
        <span>Register A Bank</span>
     </Link>
    </li>{/* End Register Page Nav */}
    
    
  </ul>
</aside>{/* End Sidebar*/}
</>
  )
}

export default SideMenu