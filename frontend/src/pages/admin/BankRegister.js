import React, { useState, useEffect } from 'react';
import PhoneInput from 'react-phone-input-2';
import 'react-phone-input-2/lib/style.css';
import { CountryFlag } from 'react-flag-kit';
import Header from '../../components/admin/Header';
import SideMenu from '../../components/admin/SideMenu';
import Footer from '../../components/admin/Footer';
import Breadcrumb from '../../components/admin/Breadcrumb';

const BankRegister = () => {
    const [formState, setFormState] = useState({
        bankName: '',
        branchAddress: '',
        ifscCode: '',
        branchState: '',
        branchCity: '',
        branchPostalCode: '',
        primaryContactName: '',
        primaryContactEmail: '',
        primaryContactPhone: '',
        loginUsername: '',
        loginPassword: '',
        authenticationMethod: '',
        employeeAccessCount: '',
        permissionLevel: '',
        terms: false,
        bankLogo: null,
        previewBankLogo: ''
    });

    const [states, setStates] = useState([]);
    const [cities, setCities] = useState([]);
    const [formErrors, setFormErrors] = useState({});

    useEffect(() => {
        fetchStates();
    }, []);

    const fetchStates = async () => {
        try {
            const response = await fetch("https://api.countrystatecity.in/v1/countries/IN/states", {
                headers: {
                    "X-CSCAPI-KEY": "aDBaTEFxcXNldnNuNXQ3YzBGWkg0QjRGSGFBYWUwaTBNUkFVY2xQTg=="
                }
            });
            if (!response.ok) {
                throw new Error('Failed to fetch states');
            }
            const data = await response.json();
            setStates(data);
        } catch (error) {
            console.error('Error fetching states:', error);
        }
    };

    const fetchCities = async (stateCode) => {
        try {
            const response = await fetch(`https://api.countrystatecity.in/v1/countries/IN/states/${stateCode}/cities`, {
                headers: {
                    "X-CSCAPI-KEY": "aDBaTEFxcXNldnNuNXQ3YzBGWkg0QjRGSGFBYWUwaTBNUkFVY2xQTg=="
                }
            });
            if (!response.ok) {
                throw new Error('Failed to fetch cities');
            }
            const data = await response.json();
            setCities(data);
        } catch (error) {
            console.error('Error fetching cities:', error);
        }
    };

    const validateForm = () => {
        const errors = {};
        if (!formState.bankName.trim()) {
            errors.bankName = 'Please enter the bank name.';
        }
        if (!formState.branchAddress.trim()) {
            errors.branchAddress = 'Please enter the branch address.';
        }
        if (!formState.ifscCode.trim() || !/^[A-Za-z]{4}[0-9]{6}$/.test(formState.ifscCode)) {
            errors.ifscCode = 'Please enter a valid IFSC code.';
        }
        if (!formState.branchState) {
            errors.branchState = 'Please select the state.';
        }
        if (!formState.branchCity) {
            errors.branchCity = 'Please select the city.';
        }
        if (!formState.branchPostalCode.trim() || !/^[0-9]{6}$/.test(formState.branchPostalCode)) {
            errors.branchPostalCode = 'Please enter a valid postal code (6 digits).';
        }
        if (!formState.primaryContactName.trim()) {
            errors.primaryContactName = 'Please enter the primary contact person\'s name.';
        }
        if (!formState.primaryContactEmail.trim() || !/\S+@\S+\.\S+/.test(formState.primaryContactEmail)) {
            errors.primaryContactEmail = 'Please enter a valid email address.';
        }
        if (!formState.loginUsername.trim()) {
            errors.loginUsername = 'Please enter a username for login credentials.';
        }
        if (!formState.loginPassword.trim()) {
            errors.loginPassword = 'Please enter a password for login credentials.';
        }
        if (!formState.primaryContactPhone.trim()) {
            errors.primaryContactPhone = 'Please enter a valid phone number.';
        }
        if (!formState.authenticationMethod) {
            errors.authenticationMethod = 'Please select an authentication method.';
        }
        if (!formState.employeeAccessCount.trim() || isNaN(formState.employeeAccessCount) || parseInt(formState.employeeAccessCount) < 0 || parseInt(formState.employeeAccessCount) > 2) {
            errors.employeeAccessCount = 'Please enter a number between 0 and 2.';
        }
        if (!formState.permissionLevel) {
            errors.permissionLevel = 'Please select a permission level.';
        }
        if (!formState.terms) {
            errors.terms = 'You must agree before submitting.';
        }
        setFormErrors(errors);
        return Object.keys(errors).length === 0;
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
    
        // Check if there are any form errors
        const isFormValid = validateForm();
    
        if (!isFormValid) {
            // If there are errors, do not submit the form
            alert('Please fill in all required fields correctly.');
            return;
        }
    
        try {
            const response = await fetch('http://localhost:5000/register_bank', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formState)
            });
    
            if (!response.ok) {
                throw new Error('Failed to register bank');
            }
    
            setFormState({
                ...formState,
                bankName: '',
                branchAddress: '',
                ifscCode: '',
                branchState: '',
                branchCity: '',
                branchPostalCode: '',
                primaryContactName: '',
                primaryContactEmail: '',
                primaryContactPhone: '',
                loginUsername: '',
                loginPassword: '',
                authenticationMethod: '',
                employeeAccessCount: '',
                permissionLevel: '',
                terms: false,
                bankLogo: null,
                previewBankLogo: ''
            });
    
            alert('Bank registration successful!');
        } catch (error) {
            console.error('Error registering bank:', error);
            alert('Failed to register bank. Please try again later.');
        }
    };
    

    const handleChange = (event) => {
        const { name, value } = event.target;
        let errorMessage = '';

        if (name === 'primaryContactEmail') {
            // Email validation regex
            const emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
            errorMessage = !emailPattern.test(value) ? 'Please enter a valid email address.' : '';
        } else if (name === 'loginPassword') {
            // Password validation regex
            const passwordPattern = /^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*#?&]).{8,}$/;
            errorMessage = !passwordPattern.test(value) ? 'Password must contain at least one uppercase letter, one lowercase letter, one number, one special character (@$!%*#?&), and be at least 8 characters long.' : '';
        }

        setFormState(prevState => ({
            ...prevState,
            [name]: value
        }));

        setFormErrors(prevErrors => ({
            ...prevErrors,
            [name]: errorMessage
        }));
    };


    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setFormState(prevState => ({
            ...prevState,
            bankLogo: file,
            previewBankLogo: URL.createObjectURL(file)
        }));
    };

    const cancelUpload = () => {
        document.getElementById('bankLogo').value = ''; // Clear the file input
        setFormState(prevState => ({
            ...prevState,
            bankLogo: null,
            previewBankLogo: ''
        }));
    };

    return (
        <>
            <Header />
            <SideMenu />
            <main id="main" className="main">
            <Breadcrumb active="BankRegister" />
                <div className="container">
                    <section className="section register min-vh-100 d-flex flex-column align-items-center justify-content-center py-4">
                        <div className="container">
                            <div className="row justify-content-center">
                                <div className="col-lg-10 col-md-6 col-12 d-flex flex-column align-items-center justify-content-center">
                                    <div className="card mb-3">
                                        <div className="card-body">
                                            <div className="pt-4 pb-2">
                                                <h5 className="card-title text-center pb-0 fs-4">Create an Account for your Bank</h5>
                                                <p className="text-center small">Enter your Bank Details to create account</p>
                                            </div>
                                            <form className="row g-3" id="registrationForm" noValidate onSubmit={handleSubmit}>
                                            <div className="mb-3">
                                                    <label htmlFor="bankLogo" className="form-label">Upload Bank Logo</label>
                                                    <div className="input-group">
                                                        <input type="file" id="bankLogo" accept="image/*" onChange={handleFileChange} />
                                                        <button className="btn" type="button" onClick={cancelUpload}>
                                                            <i className="bi bi-x" />
                                                        </button>
                                                    </div>
                                                </div>
                                                <div className="col-12">
                                                    <label htmlFor="bankName" className="form-label">Bank Name</label>
                                                    <input type="text" name="bankName" value={formState.bankName} onChange={handleChange} className={`form-control ${formErrors.bankName ? 'is-invalid' : ''}`} id="bankName" required />
                                                    <div className="invalid-feedback">{formErrors.bankName}</div>
                                                </div>

                                                <div className="col-12">
                                                    <label htmlFor="branchAddress" className="form-label">Branch Address</label>
                                                    <input type="text" name="branchAddress" value={formState.branchAddress} onChange={handleChange} className={`form-control ${formErrors.branchAddress ? 'is-invalid' : ''}`} id="branchAddress" required />
                                                    <div className="invalid-feedback">{formErrors.branchAddress}</div>
                                                </div>
                                                <div className="col-12">
                                                    <label htmlFor="ifscCode" className="form-label">IFSC Code</label>
                                                    <input type="text" name="ifscCode" className={`form-control ${formErrors.ifscCode ? 'is-invalid' : ''}`} value={formState.ifscCode} onChange={handleChange} id="ifscCode" pattern="[A-Za-z]{4}[0-9]{7}" required />
                                                    <div className="invalid-feedback">{formErrors.ifscCode}</div>
                                                </div>
                                                {/* State Field */}
                                                <div className="col-md-4">
                                                    <label htmlFor="branchState" className="form-label">State</label>
                                                    <select
                                                        name="branchState"
                                                        className={`form-select ${formErrors.branchState ? 'is-invalid' : ''}`}
                                                        id="branchState"
                                                        value={formState.branchState}
                                                        onChange={(e) => {
                                                            handleChange(e);
                                                            fetchCities(e.target.value);
                                                        }}
                                                        required
                                                    >
                                                        <option value="" disabled>Select State</option>
                                                        {states.map(state => (
                                                            <option key={state.iso2} value={state.iso2}>{state.name}</option>
                                                        ))}
                                                    </select>
                                                    <div className="invalid-feedback">{formErrors.branchState}</div>
                                                </div>
                                                {/* City Field */}
                                                <div className="col-md-4">
                                                    <label htmlFor="branchCity" className="form-label">City</label>
                                                    <select
                                                        name="branchCity"
                                                        className={`form-select ${formErrors.branchCity ? 'is-invalid' : ''}`}
                                                        id="branchCity"
                                                        value={formState.branchCity}
                                                        onChange={handleChange}
                                                        required
                                                    >
                                                        <option value="" disabled>Select City</option>
                                                        {cities.map(city => (
                                                            <option key={city.name} value={city.name}>{city.name}</option>
                                                        ))}
                                                    </select>
                                                    <div className="invalid-feedback">{formErrors.branchCity}</div>
                                                </div>
                                                <div className="col-md-4">
                                                    <label htmlFor="branchPostalCode" className="form-label">Postal Code</label>
                                                    <input type="text" name="branchPostalCode" value={formState.branchPostalCode} onChange={handleChange} className={`form-control ${formErrors.branchPostalCode ? 'is-invalid' : ''}`} id="branchPostalCode" pattern="[0-9]{6}" required />
                                                    <div className="invalid-feedback">{formErrors.branchPostalCode}</div>
                                                </div>
                                                <div className="col-md-6">
                                                    <label htmlFor="primaryContactName" className="form-label">Primary Contact Person Name</label>
                                                    <input type="text" name="primaryContactName" value={formState.primaryContactName} onChange={handleChange} className={`form-control ${formErrors.primaryContactName ? 'is-invalid' : ''}`} id="primaryContactName" required />
                                                    <div className="invalid-feedback">{formErrors.primaryContactName}</div>
                                                </div>
                                                <div className="col-md-6">
                                                    <label htmlFor="primaryContactEmail" className="form-label">Primary Contact Person Email</label>
                                                    <input
                                                        type="email"
                                                        name="primaryContactEmail"
                                                        value={formState.primaryContactEmail}
                                                        onChange={handleChange}
                                                        className={`form-control ${formErrors.primaryContactEmail ? 'is-invalid' : ''}`}
                                                        id="primaryContactEmail"
                                                        pattern="[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                                                        required
                                                    />
                                                    <div className="invalid-feedback">{formErrors.primaryContactEmail}</div>
                                                </div>

                                                <div className="col-md-6">
                                                    <label htmlFor="loginUsername" className="form-label">Login Credentials Username</label>
                                                    <input type="text" name="loginUsername" value={formState.loginUsername} onChange={handleChange} className={`form-control ${formErrors.loginUsername ? 'is-invalid' : ''}`} id="loginUsername" required />
                                                    <div className="invalid-feedback">{formErrors.loginUsername}</div>
                                                </div>
                                                <div className="col-md-6">
                                                    <label htmlFor="loginPassword" className="form-label">Login Credentials Password</label>
                                                    <input
                                                        type="password"
                                                        name="loginPassword"
                                                        value={formState.loginPassword}
                                                        onChange={handleChange}
                                                        className={`form-control ${formErrors.loginPassword ? 'is-invalid' : ''}`}
                                                        id="loginPassword"
                                                        pattern="^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*#?&]).{8,}$"
                                                        title="Password must contain at least one uppercase letter, one lowercase letter, one number, one special character (@$!%*#?&), and be at least 8 characters long."
                                                        required
                                                    />
                                                    <div className="invalid-feedback">{formErrors.loginPassword}</div>
                                                </div>

                                                {/* Primary Contact Person Phone Number */}
                                                <div className="col-md-6">
                                                    <label htmlFor="primaryContactPhone" className="form-label">Primary Contact Person Phone</label>
                                                    <PhoneInput
                                                        country={'in'}
                                                        value={formState.primaryContactPhone}
                                                        onChange={(value) => setFormState(prevState => ({ ...prevState, primaryContactPhone: value }))}
                                                        inputClass={`form-control ${formErrors.primaryContactPhone ? 'is-invalid' : ''}`}
                                                        inputProps={{
                                                            id: 'primaryContactPhone',
                                                            name: 'primaryContactPhone',
                                                            required: true,
                                                        }}
                                                    />
                                                    <div className="invalid-feedback">{formErrors.primaryContactPhone}</div>
                                                </div>
                                                {/* Authentication Method */}
                                                <div className="col-md-6">
                                                    <label htmlFor="authenticationMethod" className="form-label">Authentication Method</label>
                                                    <select name="authenticationMethod" value={formState.authenticationMethod} onChange={handleChange} className={`form-select ${formErrors.authenticationMethod ? 'is-invalid' : ''}`} id="authenticationMethod" required>
                                                        <option value="" disabled>Select authentication method</option>
                                                        <option value="accessCode">Access Code</option>
                                                        <option value="emailVerification">Email Verification</option>
                                                    </select>
                                                    <div className="invalid-feedback">{formErrors.authenticationMethod}</div>
                                                </div>
                                                {/* Employee Access Count */}
                                                <div className="col-md-6">
                                                    <label htmlFor="employeeAccessCount" className="form-label">Employee Access Count</label>
                                                    <input type="number" name="employeeAccessCount" value={formState.employeeAccessCount} onChange={handleChange} className={`form-control ${formErrors.employeeAccessCount ? 'is-invalid' : ''}`} id="employeeAccessCount" min={0} max={2} required />
                                                    <div className="invalid-feedback">{formErrors.employeeAccessCount}</div>
                                                </div>
                                                {/* Permission Level */}
                                                <div className="col-md-6">
                                                    <label htmlFor="permissionLevel" className="form-label">Permission Level</label>
                                                    <select name="permissionLevel" value={formState.permissionLevel} onChange={handleChange} className={`form-select ${formErrors.permissionLevel ? 'is-invalid' : ''}`} id="permissionLevel" required>
                                                        <option value="" disabled>Select permission level</option>
                                                        <option value="admin">Admin</option>
                                                        <option value="standardUser">Standard User</option>
                                                    </select>
                                                    <div className="invalid-feedback">{formErrors.permissionLevel}</div>
                                                </div>
                                                {/* Terms and Conditions */}
                                                <div className="col-12">
                                                    <div className="form-check">
                                                        <input className="form-check-input" name="terms" type="checkbox" checked={formState.terms} onChange={handleChange} id="acceptTerms" required />
                                                        <label className="form-check-label" htmlFor="acceptTerms">I agree and accept the <a href="#"> terms and conditions</a></label>
                                                        <div className="invalid-feedback">{formErrors.terms}</div>
                                                    </div>
                                                </div>
                                                {/* Submit Button */}
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
};

export default BankRegister;
