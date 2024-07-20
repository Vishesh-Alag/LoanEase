import React, { useState, useEffect } from 'react';
import Header from '../../components/bank/Header';
import Breadcrumb from '../../components/bank/Breadcrumb';
import SideMenu from '../../components/bank/SideMenu';
import Footer from '../../components/bank/Footer';
import axios from 'axios';
import ReactPhoneInput from 'react-phone-input-2';
import 'react-phone-input-2/lib/style.css';

const PredictLoan = () => {
    const [step, setStep] = useState(1);
    const [formData, setFormData] = useState({
        name: '',
        age: '',
        email: '',
        phone: '',
        dependents: '',
        education: '',
        employmentStatus: '',
        annualIncome: '',
        companyName: '',
        designation: '',
        location: '',
        residentialAsset: '',
        residentialAssetValue: '',
        commercialAsset: '',
        commercialAssetValue: '',
        luxuryAssets: '',
        luxuryAssetsValue: '',
        savingsAccount: '',
        currentAccount: '',
        loanAmount: '',
        loanTerm: '',
        cibilScore: '',
        bank_asset_Value: 0
    });
    const [predictionResult, setPredictionResult] = useState(null);

    useEffect(() => {
        const bankAssetValue = parseInt(formData.savingsAccount) + parseInt(formData.currentAccount);
        setFormData(prevState => ({
            ...prevState,
            bank_asset_Value: bankAssetValue
        }));
    }, [formData.savingsAccount, formData.currentAccount]);

    const nextStep = () => {
        setStep(step + 1);
    };

    const prevStep = () => {
        setStep(step - 1);
    };

    const handleChange = (e) => {
        const { name, value } = e.target;

        if (name === 'residentialAsset' || name === 'commercialAsset') {
            const assetValue = value === 'Rented' ? 0 : '';
            setFormData({
                ...formData,
                [name]: value,
                [`${name}Value`]: assetValue
            });
        } else if (name === 'luxuryAssets') {
            const assetValue = value === 'No' ? 0 : '';
            setFormData({
                ...formData,
                [name]: value,
                luxuryAssetsValue: assetValue
            });
        } else {
            setFormData({ ...formData, [name]: value });
        }
    };

    const storeUserDetailsWithResponse = async (predictionResult) => {
        try {
            // Convert specific fields to integers
            const formattedFormData = {
                ...formData,
                age: parseInt(formData.age),
                dependents: parseInt(formData.dependents),
                annualIncome: parseInt(formData.annualIncome),
                residentialAssetValue: parseInt(formData.residentialAssetValue),
                commercialAssetValue: parseInt(formData.commercialAssetValue),
                luxuryAssetsValue: parseInt(formData.luxuryAssetsValue),
                savingsAccount: parseInt(formData.savingsAccount),
                currentAccount: parseInt(formData.currentAccount),
                loanAmount: parseInt(formData.loanAmount),
                loanTerm: parseInt(formData.loanTerm),
                cibilScore: parseInt(formData.cibilScore),
                bank_asset_Value: parseInt(formData.bank_asset_Value)
            };

            // Make a POST request to store user details along with prediction response
            const userDataWithResponse = {
                ...formattedFormData,
                predictionResult: predictionResult
            };

            await axios.post('http://localhost:5000/store_LoanApp_user_details_with_response', userDataWithResponse);
        } catch (error) {
            console.error('Error:', error);
        }
    };


    const handleSubmit = async (e) => {
        e.preventDefault();
        const formattedData = {
            // Include email in the formattedData object
            email: formData.email,
            no_of_dependents: parseInt(formData.dependents),
            education: formData.education === 'Graduate' ? 1 : 0,
            self_employed: ['intern', 'workingProfessional', 'freelancer'].includes(formData.employmentStatus) ? 1 : 0,
            income_annum: parseInt(formData.annualIncome),
            loan_amount: parseInt(formData.loanAmount),
            loan_term: parseInt(formData.loanTerm),
            cibil_score: parseInt(formData.cibilScore),
            residential_assets_value: parseInt(formData.residentialAssetValue),
            commercial_assets_value: parseInt(formData.commercialAssetValue),
            luxury_assets_value: parseInt(formData.luxuryAssetsValue),
            bank_asset_value: parseInt(formData.bank_asset_Value)
        };
    
        try {
            const response = await axios.post('http://localhost:5000/predict_loan', formattedData);
            setPredictionResult(response.data);
            storeUserDetailsWithResponse(response.data);
        } catch (error) {
            console.error('Error:', error);
        }
    };
    

    const { name, age, email, phone, dependents, education, employmentStatus, annualIncome, companyName, designation, location, residentialAsset, residentialAssetValue, commercialAsset, commercialAssetValue, luxuryAssets, luxuryAssetsValue, savingsAccount, currentAccount, loanAmount, loanTerm, cibilScore, bank_asset_Value } = formData;

    const renderPredictionResult = () => {
        if (predictionResult) {
            return (
                <div>
                    <h2>Prediction Result :</h2>
                    <p>Loan Status: {predictionResult.Loan_Status}</p>
                    <p>Probability of Approval: {predictionResult.Probability_of_Approval} %</p>
                </div>
            );
        }
        return null;
    };

    return (
        <>
            <Header />
            <SideMenu />
            <main id="main" className="main">
                <Breadcrumb />
                <div>
                    {!predictionResult ? (
                        <>
                            {step === 1 && (
                                <div>
                                    <h2>Customer Personal Details</h2>
                                    <form onSubmit={nextStep}>
                                        <div>
                                            <label>Name:</label>
                                            <input type="text" name="name" value={name} onChange={handleChange} required />
                                        </div>
                                        <div>
                                            <label>Age:</label>
                                            <input type="number" name="age" value={age} onChange={handleChange} min="20" max="60" required />
                                        </div>
                                        <div>
                                            <label>Email Address:</label>
                                            <input type="email" name="email" value={email} onChange={handleChange} required />
                                        </div>
                                        <div style={{ width: '100%', maxWidth: '100%' }}>
                                            <label>Phone Number:</label>
                                            <ReactPhoneInput
                                                containerStyle={{ width: '100%' }}
                                                country={'in'}
                                                inputStyle={{ width: '100%' }}
                                                inputProps={{
                                                    name: 'phone',
                                                    required: true
                                                }}
                                                value={phone}
                                                onChange={(value) => setFormData({ ...formData, phone: value })}
                                            />
                                        </div>
                                        <div>
                                            <label>Number of Dependents:</label>
                                            <input type="number" name="dependents" value={dependents} onChange={handleChange} required />
                                        </div>
                                        <div>
                                            <label>Education:</label>
                                            <select name="education" value={education} onChange={handleChange} required>
                                                <option value="">Select</option>
                                                <option value="Graduate">Graduate</option>
                                                <option value="Not Graduate">Not Graduate</option>
                                            </select>
                                        </div>
                                        <button type="submit">Next</button>
                                    </form>
                                </div>
                            )}
                            {step === 2 && (
                                <div>
                                    <h2>Customer Income Details</h2>
                                    <form onSubmit={nextStep}>
                                        <div>
                                            <label>Employment Status:</label>
                                            <select name="employmentStatus" value={employmentStatus} onChange={handleChange} required>
                                                <option value="">Select</option>
                                                <option value="intern">Intern</option>
                                                <option value="freelancer">Freelancer</option>
                                                <option value="student">Student</option>
                                                <option value="housewife">Housewife</option>
                                                <option value="workingProfessional">Working Professional</option>
                                            </select>
                                        </div>
                                        <div>
                                            <label>Annual Income:</label>
                                            <input type="number" name="annualIncome" value={annualIncome} onChange={handleChange} required />
                                        </div>
                                        <div>
                                            <label>Company Name:</label>
                                            <input type="text" name="companyName" value={companyName} onChange={handleChange} />
                                        </div>
                                        <div>
                                            <label>Designation:</label>
                                            <input type="text" name="designation" value={designation} onChange={handleChange} />
                                        </div>
                                        <button type="button" onClick={prevStep}>Previous</button>
                                        <button type="submit">Next</button>
                                    </form>
                                </div>
                            )}
                            {step === 3 && (
                                <div>
                                    <h2>Customer Assets Details</h2>
                                    <form onSubmit={nextStep}>
                                        <div>
                                            <label>Location:</label>
                                            <select name="location" value={location} onChange={handleChange} required>
                                                <option value="">Select</option>
                                                <option value="urban">Urban</option>
                                                <option value="semiUrban">Semi-Urban</option>
                                                <option value="rural">Rural</option>
                                            </select>
                                        </div>
                                        <div>
                                            <label>Residential Asset:</label>
                                            <select name="residentialAsset" value={residentialAsset} onChange={handleChange} required>
                                                <option value="">Select</option>
                                                <option value="SelfOwned">Self-Owned</option>
                                                <option value="Rented">Rented</option>
                                            </select>
                                            {residentialAsset === "SelfOwned" && (
                                                <input type="number" name="residentialAssetValue" value={residentialAssetValue} onChange={handleChange} placeholder="Residential Asset Value" required />
                                            )}
                                        </div>
                                        <div>
                                            <label>Commercial Asset:</label>
                                            <select name="commercialAsset" value={commercialAsset} onChange={handleChange} required>
                                                <option value="">Select</option>
                                                <option value="SelfOwned">Self-Owned</option>
                                                <option value="Rented">Rented</option>
                                            </select>
                                            {commercialAsset === "SelfOwned" && (
                                                <input type="number" name="commercialAssetValue" value={commercialAssetValue} onChange={handleChange} placeholder="Commercial Asset Value" required />
                                            )}
                                        </div>
                                        <div>
                                            <label>Do you have luxury assets?</label>
                                            <select name="luxuryAssets" value={luxuryAssets} onChange={handleChange} required>
                                                <option value="">Select</option>
                                                <option value="Yes">Yes</option>
                                                <option value="No">No</option>
                                            </select>
                                            {luxuryAssets === "Yes" && (
                                                <input type="number" name="luxuryAssetsValue" value={luxuryAssetsValue} onChange={handleChange} placeholder="Luxury Assets Value" required />
                                            )}
                                        </div>
                                        <div>
                                            <label>Savings Account Value:</label>
                                            <input type="number" name="savingsAccount" value={savingsAccount} onChange={handleChange} required />
                                        </div>
                                        <div>
                                            <label>Current Account Value:</label>
                                            <input type="number" name="currentAccount" value={currentAccount} onChange={handleChange} required />
                                        </div>
                                        <button type="button" onClick={prevStep}>Previous</button>
                                        <button type="submit">Next</button>
                                    </form>
                                </div>
                            )}
                            {step === 4 && (
                                <div>
                                    <h2>Demanded Credit Details</h2>
                                    <form onSubmit={handleSubmit}>
                                        <div>
                                            <label>Loan Amount:</label>
                                            <input type="number" name="loanAmount" value={loanAmount} onChange={handleChange} required />
                                        </div>
                                        <div>
                                            <label>Loan Term:</label>
                                            <input type="number" name="loanTerm" value={loanTerm} onChange={handleChange} required />
                                        </div>
                                        <div>
                                            <label>CIBIL Score:</label>
                                            <input type="number" name="cibilScore" value={cibilScore} onChange={handleChange} required />
                                        </div>
                                        <button type="button" onClick={prevStep}>Previous</button>
                                        <button type="submit">Submit</button>
                                    </form>
                                </div>
                            )}
                        </>
                    ) : null}
                    {renderPredictionResult()}
                </div>
            </main>
            <Footer />
        </>
    );
};

export default PredictLoan;
