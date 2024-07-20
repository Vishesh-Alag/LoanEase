import React, { useState, useEffect } from 'react';
import Header from '../../components/admin/Header'
import SideMenu from '../../components/admin/SideMenu'
import Breadcrumb from '../../components/admin/Breadcrumb'
import Footer from '../../components/admin/Footer'
import { useCookies } from 'react-cookie';
import axios from 'axios'; // Import axios for making HTTP requests

const Profile = () => {
    const [cookies] = useCookies(['adminUsername']);
    const [adminData, setAdminData] = useState(null);
    const [currentPassword, setCurrentPassword] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [reenteredPassword, setReenteredPassword] = useState('');
    const [passwordMatchError, setPasswordMatchError] = useState(false);
    const [passwordChangeSuccess, setPasswordChangeSuccess] = useState(false);

    useEffect(() => {
        // Define the username to be passed to the API
        const username = cookies.adminUsername; // Replace 'your_username_here' with the actual username

        // Make an HTTP GET request to fetch admin data by username
        axios.get(`http://localhost:5000/get_admin_by_username?username=${username}`)
            .then(response => {
                // If the request is successful, set the admin data in the state
                setAdminData(response.data);
                console.log('Fetched admin data:', response.data);
            })
            .catch(error => {
                // If there is an error, log the error to the console
                console.error('Error fetching admin data:', error);
            });
    }, []); // Empty dependency array to ensure the effect runs only once

    const handleChangePassword = (e) => {
        e.preventDefault();
        if (newPassword !== reenteredPassword) {
            setPasswordMatchError(true);
            return;
        }

        axios.patch(`http://localhost:5000/change_admin_password/${cookies.adminUsername}`, {
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
            <Breadcrumb active="Profile" />
                {/* End Page Title */}
                <section className="section profile">
                    <div className="row">
                        <div className="col-xl-4">
                            <div className="card">
                                <div className="card-body profile-card pt-4 d-flex flex-column align-items-center">
                                   {/* <img src="/assets/img/profile-img.jpg" alt="Profile" className="rounded-circle" />*/}
                                    <h2>{adminData && adminData.username}</h2>
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
                                                <div className="col-lg-3 col-md-4 label ">User Name</div>
                                                <div className="col-lg-9 col-md-8">{adminData && adminData.username}</div>
                                            </div>

                                            <div className="row">
                                                <div className="col-lg-3 col-md-4 label ">Full Name</div>
                                                <div className="col-lg-9 col-md-8">{adminData && adminData.name}</div>
                                            </div>
                                            <div className="row">
                                                <div className="col-lg-3 col-md-4 label">Email</div>
                                                <div className="col-lg-9 col-md-8">{adminData && adminData.email}</div>
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
                                                    <label htmlFor="reenterPassword" className="col-md-4 col-lg-3 col-form-label">Re-enter New Password</label>
                                                    <div className="col-md-8 col-lg-9">
                                                        <input name="reenterPassword" type="password" className="form-control" value={reenteredPassword} onChange={(e) => setReenteredPassword(e.target.value)} />
                                                    </div>
                                                </div>
                                                {passwordMatchError && <div className="alert alert-danger" role="alert">Passwords do not match!</div>}
                                                {passwordChangeSuccess && <div className="alert alert-success" role="alert">Password changed successfully!</div>}
                                                <div className="text-center">
                                                    <button type="submit" className="btn btn-primary">Change Password</button>
                                                </div>
                                            </form>

                                            {/* End Change Password Form */}
                                        </div>
                                    </div>{/* End Bordered Tabs */}
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

export default Profile