import Header from '../../components/bank/Header'
import Breadcrumb from '../../components/bank/Breadcrumb'
import SideMenu from '../../components/bank/SideMenu'
import Footer from '../../components/bank/Footer'
import { useEffect, useState } from 'react';
import axios from 'axios'; // Import Axios for making HTTP requests
import '../../assets/css/CreditRiskForm.css';
import 'react-phone-input-2/lib/style.css';
import PhoneInput from 'react-phone-input-2';

// Mapping of labels to numerical representations
const labelMappings = {
	"person_home_ownership": { 'OWN': 0, 'MORTGAGE': 1, 'RENT': 2, 'OTHER': 3 },
	"loan_intent": { 'EDUCATION': 0, 'MEDICAL': 1, 'VENTURE': 2, 'PERSONAL': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5 },
	"cb_person_default_on_file": { 'No': 0, 'Yes': 1 }
};

const PredictCreditRisk = () => {
	const [currentTab, setCurrentTab] = useState(0);
	const [responseData, setResponseData] = useState(null);// State to hold API response data
	const [phone, setPhone] = useState(''); 


	useEffect(() => {
		showTab(currentTab);
	}, [currentTab]);

	const showTab = (n) => {
		const tabs = document.querySelectorAll('.tab');
		tabs.forEach((tab, index) => {
			tab.style.display = index === n ? 'block' : 'none';
		});

		const prevBtn = document.getElementById('prevBtn');
		const nextBtn = document.getElementById('nextBtn');

		prevBtn.style.display = n === 0 ? 'none' : 'inline';
		nextBtn.innerHTML = n === tabs.length - 1 ? 'Submit' : 'Next';

		fixStepIndicator(n);
	};

	const nextPrev = (n) => {
		const tabs = document.querySelectorAll('.tab');
		if (n === 1 && !validateForm()) {
			return false;
		}
		tabs[currentTab].style.display = 'none';
		setCurrentTab(currentTab + n);
		if (currentTab === tabs.length - 1 && n === 1) {
			// Submit the form data only if it's the last tab and 'Next' button is clicked
			handleSubmit();


		}
	};

	const validateForm = () => {
		const currentTabInputs = document.querySelectorAll('.tab')[currentTab].querySelectorAll('input[required], select[required]');
		let valid = true;

		currentTabInputs.forEach((input) => {
			input.classList.remove('invalid'); // Remove previous invalid class
			let fieldName;

			// Check if the input has a placeholder attribute or a name attribute
			if (input.getAttribute('placeholder')) {
				fieldName = input.getAttribute('placeholder');
			} else if (input.getAttribute('name')) {
				fieldName = input.getAttribute('name');
			} else {
				fieldName = 'Field';
			}

			// Try to find the label associated with the input
			const label = input.closest('label');
			if (label) {
				const labelText = label.textContent.trim();
				if (labelText) {
					fieldName = labelText;
				}
			}

			// Validate numeric fields
			if (input.type === 'number') {
				let numericValue = parseFloat(input.value);
				if (isNaN(numericValue) || numericValue < 0) {
					input.classList.add('invalid');
					alert(`${fieldName}"must be a non-negative numeric value."` );
					numericValue = 0; // Set negative values to 0
					input.value = numericValue; // Update input value
					valid = false;
				}
			}

			// Validate selector fields
			if (input.tagName.toLowerCase() === 'select') {
				if (input.value === '') {
					input.classList.add('invalid');
					alert(`Please select an option for ${fieldName}.`);
					valid = false;
				}
			}

			// Validate age
			if (input.id === 'personAge') {
				const age = parseInt(input.value);
				if (isNaN(age) || age < 18 || age > 70) {
					input.classList.add('invalid');
					alert(`${fieldName} must be in the range from 18 to 70.`);
					valid = false;
				}
			}

			// Validate person_emp_length
			if (input.id === 'personEmpLength') {
				const empLength = parseFloat(input.value);
				if (isNaN(empLength) || empLength < 0 || empLength > 50) {
					input.classList.add('invalid');
					alert(`${fieldName} must be between 0 to 50.`);
					valid = false;
				}
			}

			// Validate email format
			if (input.type === 'email') {
				const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
				if (!emailPattern.test(input.value)) {
					input.classList.add('invalid');
					alert(`Please enter a valid email address for ${fieldName}.`);
					valid = false;
				}
			}

			// Validate phone number
            if (input.name === 'Phone Number' && !phone) {
                input.classList.add('invalid');
                alert('Phone Number is required.');
                valid = false;
            }

			// You can add other validations here as needed
		});

		if (valid) {
			const steps = document.querySelectorAll('.step');
			steps[currentTab].classList.add('finish');
		}

		return valid;
	};



	const fixStepIndicator = (n) => {
		const steps = document.querySelectorAll('.step');
		steps.forEach((step, index) => {
			step.classList.toggle('active', index === n);
		});
	};

	const handleSubmit = async () => {
		const formData = {
			"email": document.querySelector('input[name="Email Address"]').value,
			"person_age": parseInt(document.getElementById('personAge').value),
			"person_income": parseFloat(document.getElementById('personAnnualIncome').value),
			"person_home_ownership": labelMappings["person_home_ownership"][document.getElementById('personHomeOwnership').value],
			"person_emp_length": parseFloat(document.getElementById('personEmpLength').value),
			"loan_intent": labelMappings["loan_intent"][document.getElementById('loanIntent').value],
			"loan_amnt": parseInt(document.getElementById('loanAmount').value),
			"loan_int_rate": parseFloat(document.getElementById('loanInterestRate').value),
			"loan_percent_income": parseFloat(document.getElementById('loanPercentOfIncome').value),
			"cb_person_default_on_file": labelMappings["cb_person_default_on_file"][document.getElementById('cbPersonDefaultOnFile').value],
			"cb_person_cred_hist_length": parseInt(document.getElementById('cbPersonCredHistLength').value)
		};

		try {
			const response = await axios.post('http://localhost:5000/predict_creditrisk', formData);
			if (response.status === 200) {
				setResponseData(response.data); // Set response data to state
				storeFormData(response.data);

			} else {
				throw new Error('Failed to fetch');
			}
		} catch (error) {
			console.error('Error:', error);
		}
	};

	const storeFormData = async (responseData) => {
		// Gather all the form data filled by the user
		const formData = {
			name: document.querySelector('input[name="Name"]').value,
			phone,
			email: document.querySelector('input[name="Email Address"]').value,
			person_age: parseInt(document.getElementById('personAge').value),
			person_income: parseFloat(document.getElementById('personAnnualIncome').value),
			person_home_ownership: document.getElementById('personHomeOwnership').value,
			person_emp_length: parseFloat(document.getElementById('personEmpLength').value),
			loan_intent: document.getElementById('loanIntent').value,
			loan_amnt: parseInt(document.getElementById('loanAmount').value),
			loan_int_rate: parseFloat(document.getElementById('loanInterestRate').value),
			loan_percent_income: parseFloat(document.getElementById('loanPercentOfIncome').value),
			cb_person_default_on_file: document.getElementById('cbPersonDefaultOnFile').value,
			cb_person_cred_hist_length: parseInt(document.getElementById('cbPersonCredHistLength').value),
			responseData: responseData


		};

		try {
			const response = await axios.post('http://localhost:5000/store_credit_risk_user_details', formData);
			if (response.status === 200) {
				// Handle successful storage response if needed
				console.log('Form data stored successfully:', response.data);
			} else {
				throw new Error('Failed to store form data');
			}
		} catch (error) {
			console.error('Error storing form data:', error);
		}
	};



	return (
		<>
			<Header />
			<SideMenu />
			<main id="main" className="main">
				<Breadcrumb />
				{responseData ? (
					<div className="response-container">
						<h2>Prediction Result :</h2>
						<div>
							{/* Render response in client-side format */}
							{/* Assuming responseData is an object with key-value pairs */}
							{Object.entries(responseData).map(([key, value]) => (
								<p key={key}>
									{key}: {key === 'Probability of Default' ? `${value} %` : value}
								</p>
							))}
						</div>
					</div>
				) : (

					<form id="multiStepForm" action="">

						<div className="tab">
							<h2>Customer Personal Details</h2>
							<h3>Name</h3>
							<p><input type="text" placeholder="Name" name="Name" required /></p>
							<h3>Email Address</h3>
							<p><input type="email" name="Email Address" id='Email' placeholder="Email Address" required /></p>
							<h3>Phone Number</h3>
							<p>
                                <PhoneInput
                                    country={'in'}
                                    value={phone}
                                    onChange={setPhone}
                                    inputProps={{
                                        name: 'Phone Number',
                                        required: true,
                                        autoFocus: true,
										style: { width: '100%' } // Inline CSS for the phone input field
                                    }}
                                    containerStyle={{ width: '100%' }} // Inline CSS for the container
                                    inputStyle={{ width: '100%' }} // Inline CSS for the input field
                                    buttonStyle={{ height: '100%' }} // Inline CSS for the dropdown button
                                    
                                />
                            </p>
						</div>

						<div className="tab">
							<h2>Customer Income and Employment Details</h2>
							<h3>Age of the Person</h3>
							<p><input type="number" name="age" id="personAge" placeholder="Person's Age" required /></p>
							<h3>Annual Income of the Person</h3>
							<p><input type="number" name="annual income" id='personAnnualIncome' placeholder="Person's Annual Income" required /></p>
							<h3>Home Ownership</h3>
							<p>
								<select id='personHomeOwnership' name="Person Home Ownership" defaultValue="" required>
									<option value="" disabled>Select Home Ownership</option>
									<option value="OWN">Own</option>
									<option value="MORTGAGE">Mortgage</option>
									<option value="RENT">Rent</option>
									<option value="OTHER">Other</option>
								</select>
							</p>
							<h3>Employment Length</h3>
							<p><input type="number" name="Employment length" id='personEmpLength' placeholder="Employment Length (in years)" required /></p>
						</div>

						<div className="tab">
							<h2>Loan Details</h2>
							<h3>Loan Intent</h3>
							<p>
								<select id='loanIntent' name="Loan Intent" defaultValue="" required>
									<option value="" disabled>Select Loan Intent</option>
									<option value="EDUCATION">Education</option>
									<option value="MEDICAL">Medical</option>
									<option value="VENTURE">Venture</option>
									<option value="PERSONAL">Personal</option>
									<option value="HOMEIMPROVEMENT">Home Improvement</option>
									<option value="DEBTCONSOLIDATION">Debt Consolidation</option>
								</select>
							</p>
							<h3>Loan Amount</h3>
							<p><input type="number" name="Loan Amount" id='loanAmount' placeholder="Loan Amount" required /></p>
							<h3>Loan Interest Rate</h3>
							<p><input type="number" name="Loan Interest Rate " id='loanInterestRate' placeholder="Loan Interest Rate" required /></p>
							<h3>Loan Percent of Income <h4>(Loan Amount / Person Income)</h4></h3>
							<p><input type="number" name="Loan Percent Income " id='loanPercentOfIncome' placeholder="Loan Percent of Income" required /></p>
							<h3>Deault on File</h3>
							<p>
								<select id='cbPersonDefaultOnFile' name="Default on File" defaultValue="" required>
									<option value="" disabled>Select Default on File</option>
									<option value="No">No</option>
									<option value="Yes">Yes</option>
								</select>
							</p>
							<h3>Credit History Length</h3>
							<p><input type="number" name="Credit History Length" id='cbPersonCredHistLength' placeholder="Credit History Length" required /></p>
						</div>

						<div style={{ overflow: "auto" }}>
							<div style={{ float: "right" }}>
								<button type="button" id="prevBtn" onClick={() => { nextPrev(-1) }}>Previous</button>
								<button type="button" id="nextBtn" onClick={() => { nextPrev(1) }}>Next</button>
							</div>
						</div>

						<div style={{ textAlign: "center", marginTop: "40px" }}>
							<span className="step"></span>
							<span className="step"></span>
							<span className="step"></span>
						</div>
					</form>)}
			</main>
			<Footer />
		</>
	);
};

export default PredictCreditRisk;