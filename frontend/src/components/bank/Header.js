import React, { useState, useEffect } from 'react';
import { useCookies } from 'react-cookie';
import { Link } from 'react-router-dom'; // Import Link from react-router-dom

const Header = () => {

  const [cookies] = useCookies(['username']);
  const [userData, setUserData] = useState(null);

  useEffect(() => {
    if (cookies.username) {
      fetch(`http://localhost:5000/get_bank_by_username?username=${cookies.username}`)
        .then(response => {
          if (response.ok) {
            return response.json();
          }
          throw new Error('Network response was not ok.');
        })
        .then(data => {
          setUserData(data);
        })
        .catch(error => {
          console.error('Error fetching bank data:', error);
        });
    }
  }, [cookies.username]);

  const handleLogout = () => {
    // Clear cookies by setting their expiration date to the past
    document.cookie = 'username=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
    document.cookie = 'password=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
    document.cookie = 'loggedIn=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';

    // Redirect the user to the login page
    window.location.href = '/bank/login'; // Replace '/login' with the actual URL of your login page
  };
  useEffect(() => {
    console.log('Cookies:', cookies);
  }, [cookies]);

  return (
    <>
      {/* ======= Header ======= */}
      <header id="header" className="header fixed-top d-flex align-items-center">
        <div className="d-flex align-items-center justify-content-between">
          <a href="index.html" className="logo d-flex align-items-center">
            <img src="/assets/img/logo.png" alt />
            <span className="d-none d-lg-block">{userData ? userData.bankName : 'Loading...'}</span>
          </a>
        </div>{/* End Logo */}
        <nav className="header-nav ms-auto">
          <ul className="d-flex align-items-center">
           
            <li className="nav-item dropdown pe-3">
              <a className="nav-link nav-profile d-flex align-items-center pe-0" href="#" data-bs-toggle="dropdown">
                {/*<img src="/assets/img/profile-img.jpg" alt="Profile" className="rounded-circle" />*/}
                <span className="d-none d-md-block dropdown-toggle ps-2">{cookies.username}</span>
              </a>{/* End Profile Iamge Icon */}
              <ul className="dropdown-menu dropdown-menu-end dropdown-menu-arrow profile">
                <li className="dropdown-header">
                  <h6>{cookies.username}</h6>
                </li>
                <li>
                  <hr className="dropdown-divider" />
                </li>
                <li>
                <Link to="/bank/profile" className="dropdown-item d-flex align-items-center">
                    <i className="bi bi-person" />
                    <span>My Profile</span>
                  </Link>
                </li>
                <li>
                  <hr className="dropdown-divider" />
                </li>
                <li>
                  <hr className="dropdown-divider" />
                </li>
                <li>
                  <a className="dropdown-item d-flex align-items-center" href="#" onClick={handleLogout}>
                    <i className="bi bi-box-arrow-right" />
                    <span>Sign Out</span>
                  </a>
                </li>
              </ul>{/* End Profile Dropdown Items */}
            </li>{/* End Profile Nav */}
          </ul>
        </nav>{/* End Icons Navigation */}
      </header>{/* End Header */}
    </>
  )
}

export default Header