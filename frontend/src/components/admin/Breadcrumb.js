import React from 'react';

const Breadcrumb = ({ active }) => {
  return (
    <div className="pagetitle">
      <nav>
        <ol className="breadcrumb">

          {active === 'Dashboard' && (
            <h1><li className="breadcrumb-item active">Dashboard</li></h1>
          )}
          {active === 'Profile' && (
            <h1> <li className="breadcrumb-item active">Profile</li></h1>
          )}
          {active === 'BankRegister' && (
            <h1><li className="breadcrumb-item active">BankRegister</li></h1>
          )}

          {active === 'Register' && (
            <h1><li className="breadcrumb-item active">Register Admin</li></h1>
          )}
        </ol>
      </nav>
    </div>
  );
}

export default Breadcrumb;
