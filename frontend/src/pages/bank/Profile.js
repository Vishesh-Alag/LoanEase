import React, { useState, useEffect } from 'react';
import Header from '../../components/bank/Header'
import SideMenu from '../../components/bank/SideMenu'
import Breadcrumb from '../../components/bank/Breadcrumb'
import Footer from '../../components/bank/Footer'
import { useCookies } from 'react-cookie';
import axios from 'axios'; // Import axios for making HTTP requests


const Profile = () => {
    const [cookies] = useCookies(['username']);
    const [userData, setUserData] = useState(null);
    const [currentPassword, setCurrentPassword] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [reenteredPassword, setReenteredPassword] = useState('');
    const [passwordMatchError, setPasswordMatchError] = useState(false);
    const [passwordChangeSuccess, setPasswordChangeSuccess] = useState(false);

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
                    console.log(userData)
                })
                .catch(error => {
                    console.error('Error fetching bank data:', error);
                });
        }
    }, [cookies.username]);

    const handleChangePassword = (e) => {
        e.preventDefault();
        if (newPassword !== reenteredPassword) {
            setPasswordMatchError(true);
            return;
        }

        axios.patch(`http://localhost:5000/change_bank_password/${cookies.username}`, {
            current_password: currentPassword,
            new_password: newPassword
        })
            .then(response => {
                setNewPassword('');
                setReenteredPassword('');
                setCurrentPassword('');
                setPasswordMatchError(false);
                setPasswordChangeSuccess(true);
                setTimeout(() => setPasswordChangeSuccess(false), 10000);
            })
            .catch(error => {
                console.error('Error changing password:', error);
            });
    };


    return (
        <>
            <Header />
            <SideMenu />
            <main id="main" className="main">
                <Breadcrumb />
                <div className="pagetitle">
                    <h1>Profile</h1>
                </div>
                <section className="section profile">
                    <div className="row">
                        <div className="col-xl-4">
                            <div className="card">
                                <div className="card-body profile-card pt-4 d-flex flex-column align-items-center">
                                    {/*<img src="/assets/img/profile-img.jpg" alt="Profile" className="rounded-circle" />*/}
                                    <h2>{userData ? userData.loginUsername : 'Loading...'}</h2>

                                    <div className="social-links mt-2">
                                        <a href="#" className="twitter"><i className="bi bi-twitter" /></a>
                                        <a href="#" className="facebook"><i className="bi bi-facebook" /></a>
                                        <a href="#" className="instagram"><i className="bi bi-instagram" /></a>
                                        <a href="#" className="linkedin"><i className="bi bi-linkedin" /></a>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="col-xl-8">
                            <div className="card">
                                <div className="card-body pt-3">
                                    {/* Bordered Tabs */}
                                    <ul className="nav nav-tabs nav-tabs-bordered">
                                        <li className="nav-item">
                                            <button className="nav-link active" data-bs-toggle="tab" data-bs-target="#profile-overview">Overview</button>
                                        </li>

                                        <li className="nav-item">
                                            <button className="nav-link" data-bs-toggle="tab" data-bs-target="#profile-change-password">Change Password</button>
                                        </li>
                                    </ul>
                                    <div className="tab-content pt-2">
                                        <div className="tab-pane fade show active profile-overview" id="profile-overview">
                                            <h5 className="card-title">Profile Details</h5>
                                            <div className="row">
                                                <div className="col-lg-3 col-md-4 label">Full Name</div>
                                                <div className="col-lg-9 col-md-8">{userData ? userData.primaryContactName : 'Loading...'}</div>
                                            </div>
                                            <div className="row">
                                                <div className="col-lg-3 col-md-4 label">Bank Name</div>
                                                <div className="col-lg-9 col-md-8">{userData ? userData.bankName : 'Loading...'}</div>
                                            </div>
                                            <div className="row">
                                                <div className="col-lg-3 col-md-4 label">Branch Address</div>
                                                <div className="col-lg-9 col-md-8">{userData ? userData.branchAddress : 'Loading...'}</div>
                                            </div>
                                            <div className="row">
                                                <div className="col-lg-3 col-md-4 label">IFSC Code</div>
                                                <div className="col-lg-9 col-md-8">{userData ? userData.ifscCode : 'Loading...'}</div>
                                            </div>
                                            <div className="row">
                                                <div className="col-lg-3 col-md-4 label">Branch State</div>
                                                <div className="col-lg-9 col-md-8">{userData ? userData.branchState : 'Loading...'}</div>
                                            </div>
                                            <div className="row">
                                                <div className="col-lg-3 col-md-4 label">Branch City</div>
                                                <div className="col-lg-9 col-md-8">{userData ? userData.branchCity : 'Loading...'}</div>
                                            </div>
                                            <div className="row">
                                                <div className="col-lg-3 col-md-4 label">Branch Postal Code</div>
                                                <div className="col-lg-9 col-md-8">{userData ? userData.branchPostalCode : 'Loading...'}</div>
                                            </div>
                                            <div className="row">
                                                <div className="col-lg-3 col-md-4 label">Primary Contact Email</div>
                                                <div className="col-lg-9 col-md-8">{userData ? userData.primaryContactEmail : 'Loading...'}</div>
                                            </div>
                                            <div className="row">
                                                <div className="col-lg-3 col-md-4 label">Primary Contact Phone</div>
                                                <div className="col-lg-9 col-md-8">{userData ? userData.primaryContactPhone : 'Loading...'}</div>
                                            </div>
                                        </div>

                                        <div className="tab-pane fade pt-3" id="profile-change-password">
                                            {/* Change Password Form */}
                                            <form onSubmit={handleChangePassword}>
                                                <div className="row mb-3">
                                                    <label htmlFor="currentPassword" className="col-md-4 col-lg-3 col-form-label">Current Password</label>
                                                    <div className="col-md-8 col-lg-9">
                                                        <input name="currentPassword" type="password" className="form-control" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} />
                                                    </div>
                                                </div>
                                                <div className="row mb-3">
                                                    <label htmlFor="newPassword" className="col-md-4 col-lg-3 col-form-label">New Password</label>
                                                    <div className="col-md-8 col-lg-9">
                                                        <input name="newPassword" type="password" className="form-control" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} />
                                                    </div>
                                                </div>
                                                <div className="row mb-3">
                                                    <label htmlFor="reenteredPassword" className="col-md-4 col-lg-3 col-form-label">Re-enter New Password</label>
                                                    <div className="col-md-8 col-lg-9">
                                                        <input name="reenteredPassword" type="password" className="form-control" value={reenteredPassword} onChange={(e) => setReenteredPassword(e.target.value)} />
                                                    </div>
                                                </div>
                                                {passwordMatchError && <div className="alert alert-danger" role="alert">Passwords do not match!</div>}
                                                {passwordChangeSuccess && <div className="alert alert-success" role="alert">Password changed successfully!</div>}
                                                <div className="text-center">
                                                    <button type="submit" className="btn btn-primary">Change Password</button>
                                                </div>
                                            </form>{/* End Change Password Form */}
                                        </div>{/* End Bordered Tabs */}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>
            </main>{/* End #main */}
            <Footer />
        </>
    )
}


export default Profile;
