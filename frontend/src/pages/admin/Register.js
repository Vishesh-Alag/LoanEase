import React, { useState, useRef } from 'react';
import Header from '../../components/admin/Header';
import Footer from '../../components/admin/Footer';
import Breadcrumb from '../../components/admin/Breadcrumb';
import SideMenu from '../../components/admin/SideMenu';

const Register = () => {
  const initialFormData = {
    name: '',
    email: '',
    username: '',
    password: ''
  };
  const [formData, setFormData] = useState(initialFormData)
  const [showPasswordHint, setShowPasswordHint] = useState(false);
  const passwordInputRef = useRef(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value
    }));
    if (name === 'password') {
      setShowPasswordHint(!e.target.checkValidity());
      
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const form = e.target;
    if (form.checkValidity()) {
      try {
        const response = await fetch('http://localhost:5000/register_admin', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(formData)
        });
        if (!response.ok) {
          throw new Error('Failed to register');
        }
        else{
        setFormData(initialFormData);
        alert('Account created successfully!');
        }
      } catch (error) {
        console.error('Error registering:', error);
        alert('Failed to create account. Please try again.');
      }
    } else {
      form.reportValidity();
      setShowPasswordHint(true);
      
    }
  };

  return (
    <>
      <Header />
      <SideMenu />
      <main id="main" className='main'>
      <Breadcrumb active="Register" />
        <div className="container">
          <section className="section register">
            <div className="container">
              <div className="row justify-content-center">
                <div className="col-lg-4 col-md-6 d-flex flex-column align-items-center justify-content-center">
                  <div className="card">
                    <div className="card-body">
                      <div>
                        <h5 className="card-title text-center pb-0 fs-4">Create an Admin Account</h5>
                        <p className="text-center small">Enter your personal details to create account</p>
                      </div>
                      <form className="row g-3" onSubmit={handleSubmit}>
                        <div className="col-12">
                          <label htmlFor="yourName" className="form-label">Your Name</label>
                          <input type="text" name="name" className="form-control" id="yourName" value={formData.name} onChange={handleChange} required />
                          <div className="invalid-feedback">Please enter your name!</div>
                        </div>
                        <div className="col-12">
                          <label htmlFor="yourEmail" className="form-label">Your Email</label>
                          <input type="email" name="email" className="form-control" id="yourEmail" value={formData.email} onChange={handleChange} required />
                          <div className="invalid-feedback">Please enter a valid Email address!</div>
                        </div>
                        <div className="col-12">
                          <label htmlFor="yourUsername" className="form-label">Username</label>
                          <div className="input-group">
                            <span className="input-group-text" id="inputGroupPrepend">@</span>
                            <input type="text" name="username" className="form-control" id="yourUsername" value={formData.username} onChange={handleChange} required />
                            <div className="invalid-feedback">Please choose a username.</div>
                          </div>
                        </div>
                        <div className="col-12">
                          <label htmlFor="yourPassword" className="form-label">Password</label>
                          <input type="password" name="password" className="form-control" id="yourPassword" value={formData.password} onChange={handleChange} required pattern="^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$" ref={passwordInputRef} />
                          <div className="invalid-feedback">Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one number, and one special character (@$!%*#?&).</div>
                          {showPasswordHint && (
                            <small className="form-text text-muted">Password should contain at least 8 characters including one uppercase letter, one lowercase letter, one number, and one special character (@$!%*#?&).</small>
                          )}
                        </div>
                        <div className="col-12">
                          <button className="btn btn-primary w-100" type="submit">Create Account</button>
                        </div>
                      </form>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>
      <Footer />
    </>
  );
}

export default Register;
