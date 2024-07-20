import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useCookies } from 'react-cookie';

const Login = () => {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    remember: false
  });
  const [cookies, setCookie] = useCookies(['adminLoggedIn', 'username']);
  const [validationErrors, setValidationErrors] = useState({});

  const handleChange = (e) => {
    const { name, value, checked } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: name === 'remember' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const form = e.target;
    if (form.checkValidity()) {
      try {
        const response = await fetch('http://localhost:5000/admin-login', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(formData)
        });
        if (!response.ok) {
          throw new Error('Failed to login');
        }
        // Handle successful login
        const userData = await response.json();
        // Set cookies for loggedIn status and username
        setCookie('adminLoggedIn', true, { path: '/' });
        setCookie('adminUsername', formData.username, { path: '/'  });
        // Redirect to dashboard
        window.location.href = '/admin/dashboard';
      } catch (error) {
        console.error('Error logging in:', error);
        alert('Failed to login. Please try again.');
      }
    } else {
      setValidationErrors({
        ...Object.fromEntries(
          Array.from(form.elements).map(element => [element.name, element.validationMessage])
        )
      });
    }
  };

  useEffect(() => {
    console.log('Username cookie value:', cookies.username);
    if (cookies.adminLoggedIn) {
      window.location.href = '/admin/dashboard'; // Redirect if logged in
    }
  }, [cookies.adminLoggedIn]);
  return (
    <>
      <main>
        <div className="container">
          <section className="section register min-vh-100 d-flex flex-column align-items-center justify-content-center py-4">
            <div className="container">
              <div className="row justify-content-center">
                <div className="col-lg-4 col-md-6 d-flex flex-column align-items-center justify-content-center">
                  <div className="d-flex justify-content-center py-4">
                    <Link to="/" className="logo d-flex align-items-center w-auto">
                      <img src="/assets/img/logo.png" alt="Logo" />
                      <span className="d-none d-lg-block">LoanEase</span>
                    </Link>
                  </div>
                  <div className="card mb-3">
                    <div className="card-body">
                      <div className="pt-4 pb-2">
                        <h5 className="card-title text-center pb-0 fs-4">Login to Your Admin Account</h5>
                        <p className="text-center small">Enter your username &amp; password to login</p>
                      </div>
                      <form className="row g-3" noValidate onSubmit={handleSubmit}>
                        <div className="col-12">
                          <label htmlFor="yourUsername" className="form-label">Username</label>
                          <div className="input-group">
                            <span className="input-group-text" id="inputGroupPrepend">@</span>
                            <input type="text" name="username" className={`form-control ${validationErrors.username ? 'is-invalid' : ''}`} id="yourUsername" value={formData.username} onChange={handleChange} required />
                            <div className="invalid-feedback">{validationErrors.username}</div>
                          </div>
                        </div>
                        <div className="col-12">
                          <label htmlFor="yourPassword" className="form-label">Password</label>
                          <input type="password" name="password" className={`form-control ${validationErrors.password ? 'is-invalid' : ''}`} id="yourPassword" value={formData.password} onChange={handleChange} required />
                          <div className="invalid-feedback">{validationErrors.password}</div>
                        </div>
                        <div className="col-12">
                          <div className="form-check">
                            <input className="form-check-input" type="checkbox" name="remember" checked={formData.remember} onChange={handleChange} id="rememberMe" />
                            <label className="form-check-label" htmlFor="rememberMe">Remember me</label>
                          </div>
                        </div>
                        <div className="col-12">
                          <button className="btn btn-primary w-100" type="submit">Login</button>
                        </div>
                        
                      </form>
                    </div>
                  </div>
                  <div className="credits">
                    Powered by <a href="/">LoanEase</a>
                    , Developed by <a href="/">Vishesh Alag</a>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>
    </>
  );
}

export default Login;
