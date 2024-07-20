  import React, { useState } from 'react';
  import Header from '../../components/bank/Header';
  import Breadcrumb from '../../components/bank/Breadcrumb';
  import SideMenu from '../../components/bank/SideMenu';
  import Footer from '../../components/bank/Footer';
  import axios from 'axios';

  const Contact = () => {
    const initialFormData = {
      name: '',
      email: '',
      subject: '',
      message: ''
    };
  
    const [formData, setFormData] = useState(initialFormData);
    const [messageStatus, setMessageStatus] = useState('');
  
    const handleChange = (e) => {
      const { name, value } = e.target;
      setFormData({
        ...formData,
        [name]: value
      });
    };
  
    const handleSubmit = async (e) => {
      e.preventDefault();
      try {
        const response = await axios.post('http://localhost:5000/store_query', formData);
        if (response.status === 200) {
          setMessageStatus('Your message has been sent. Thank you!');
          // Reset the form after sending the message
          setFormData(initialFormData);

        }
      } catch (error) {
        setMessageStatus('Failed to send message. Please try again later.');
        console.error('Error:', error);
      }
    };
  
  
    return (
      <>
        <Header />
        <SideMenu />
        <main id="main" className="main">
          <Breadcrumb />
          <section className="section contact">
            <div className="row gy-4">
              <div className="col-xl-6">
                <div className="row">
                  <div className="col-lg-6">
                    <div className="info-box card">
                      <i className="bi bi-geo-alt" />
                      <h3>Address</h3>
                      <p>New Delhi,<br />India</p>
                    </div>
                  </div>
                  <div className="col-lg-6">
                    <div className="info-box card">
                      <i className="bi bi-telephone" />
                      <h3>Call Us</h3>
                      <p>+91 9289426060</p>
                    </div>
                  </div>
                  <div className="col-lg-6">
                    <div className="info-box card">
                      <i className="bi bi-envelope" />
                      <h3>Email Us</h3>
                      <p>loanease.connect@gmail.com<br />vishesh.alag03@gmail.com</p>
                    </div>
                  </div>
                  <div className="col-lg-6">
                    <div className="info-box card">
                      <i className="bi bi-clock" />
                      <h3>Open Hours</h3>
                      <p>Monday - Friday<br />11:00AM - 06:00PM</p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="col-xl-6">
                <div className="card p-4">
                  <form onSubmit={handleSubmit} className="php-email-form">
                    <div className="row gy-4">
                      <div className="col-md-6">
                        <input type="text" name="name" className="form-control" placeholder="Your Name" onChange={handleChange} value={formData.name} required />
                      </div>
                      <div className="col-md-6">
                        <input type="email" className="form-control" name="email" placeholder="Your Email" onChange={handleChange} value={formData.email} required />
                      </div>
                      <div className="col-md-12">
                        <input type="text" className="form-control" name="subject" placeholder="Subject" onChange={handleChange} value={formData.subject} required />
                      </div>
                      <div className="col-md-12">
                        <textarea className="form-control" name="message" rows={6} placeholder="Message" onChange={handleChange} value={formData.message} required />
                      </div>
                      <div className="col-md-12 text-center">
                        <div className="loading">Loading</div>
                        <div className="error-message" />
                        <div className={`sent-message alert alert-success ${messageStatus ? 'show' : 'd-none'}`}>{messageStatus}</div>
                      <button type="submit">Send Message</button>
                      </div>
                    </div>
                  </form>
                </div>
              </div>
            </div>
          </section>
        </main>
        <Footer />
      </>
    );
  };

  export default Contact;
