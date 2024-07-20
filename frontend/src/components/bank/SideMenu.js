import React from 'react'
import { Link } from 'react-router-dom'

const SideMenu = () => {
  return (
    <>
    {/* ======= Sidebar ======= */}
<aside id="sidebar" className="sidebar">
  <ul className="sidebar-nav" id="sidebar-nav">
    <li className="nav-item">
      <Link className="nav-link collapsed " to="/bank/dashboard">
        <i className="bi bi-grid" />
        <span>Dashboard</span>
      </Link>
    </li>{/* End Dashboard Nav */}
    <li className="nav-item">
      <Link className="nav-link collapsed" to="/bank/profile">
        <i className="bi bi-person" />
        <span>Profile</span>
      </Link>
    </li>{/* End Profile Page Nav */}

    <li className="nav-item">
      <Link className="nav-link collapsed" to="/bank/predict">
        <i className="bi bi-cash-coin" />
        <span>Predict Loan Approval </span>
      </Link>
    </li>{/* End Profile Page Nav */}
    <li className="nav-item">
      <Link className="nav-link collapsed" to="/bank/predict_creditrisk">
        <i className="ri-bank-card-fill" />
        <span>Predict Credit Risk </span>
      </Link>
    </li>{/* End Profile Page Nav */}
   
    <li className="nav-item">
      <Link className="nav-link collapsed" to="/bank/faq">
        <i className="bi bi-question-circle" />
        <span>F.A.Q</span>
      </Link>
    </li>{/* End Profile Page Nav */}
   
    <li className="nav-item">
      <Link className="nav-link collapsed" to="/bank/contact">
        <i className="bi bi-card-list" />
        <span>Contact</span>
     </Link>
    </li>{/* End Register Page Nav */}
    
    
  </ul>
</aside>{/* End Sidebar*/}
</>
  )
}

export default SideMenu