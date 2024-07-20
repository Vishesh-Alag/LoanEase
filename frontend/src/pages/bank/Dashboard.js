import React, { useState, useEffect } from 'react';
import Header from '../../components/bank/Header';
import Breadcrumb from '../../components/bank/Breadcrumb';
import SideMenu from '../../components/bank/SideMenu';
import Footer from '../../components/bank/Footer';
import { useCookies } from 'react-cookie';
import DataTable from 'react-data-table-component';
import Chart from 'chart.js/auto';

const Dashboard = () => {
  const [cookies] = useCookies(['loggedIn']);
  const [loanUserData, setLoanUserData] = useState([]);
  const [creditRiskUserData, setCreditRiskUserData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loanSearchText, setLoanSearchText] = useState(''); // Separate state for loan search
  const [creditRiskSearchText, setCreditRiskSearchText] = useState(''); // Separate state for credit risk search
  const [filteredLoanData, setFilteredLoanData] = useState([]);
  const [filteredCreditRiskData, setFilteredCreditRiskData] = useState([]);
  //charts
  const [approvalData, setApprovalData] = useState({});
  const [creditRiskData, setCreditRiskData] = useState({});
  const [educationData, setEducationData] = useState({});
  const [selfEmployedData, setSelfEmployedData] = useState({});
  const [approvalHistogramData, setApprovalHistogramData] = useState({});
  const [creditRiskHistogramData, setCreditRiskHistogramData] = useState({});
  const [cibilScoreData, setCibilScoreData] = useState({});
  const [ageDistributionData, setAgeDistributionData] = useState({}); // Added
  

  useEffect(() => {
    // Fetch data from both APIs
    Promise.all([
      fetch('http://localhost:5000/loan_approval_user_details'),
      fetch('http://localhost:5000/credit_risk_user_details')
    ])
      .then(([loanResponse, creditResponse]) => Promise.all([loanResponse.json(), creditResponse.json()]))
      .then(([loanData, creditData]) => {
        setLoanUserData(loanData);
        setCreditRiskUserData(creditData);
        setFilteredLoanData(loanData);
        setFilteredCreditRiskData(creditData);
        setLoading(false);
        console.log('Fetched loan approval data:', loanData);
        console.log('Fetched credit risk data:', creditData);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
        setLoading(false);
      });

      fetchApprovalData();
    fetchCreditRiskData();
    fetchEducationData();
    fetchSelfEmployedData();
    fetchApprovalHistogramData();
    fetchCreditRiskHistogramData();
    fetchCibilScoreData();
    fetchAgeDistributionData();
  }, []);

  // If the loggedIn cookie is not set, redirect to the login page
  if (!cookies.loggedIn) {
    window.location.href = '/bank/login';
    return null; // Return null to prevent rendering anything else
  }

  const handleLoanSearch = (event) => {
    const value = event.target.value.toLowerCase();
    setLoanSearchText(value);
  
    const filtered = loanUserData.filter(item =>
      item.name.toLowerCase().includes(value) ||
      item.email.toLowerCase().includes(value) ||
      (item.phone && item.phone.includes(value)) || // Check if phone exists and includes the search value
      item.age.toString().includes(value) || // Convert age to string and check if it includes the search value
      item.education.toLowerCase().includes(value) ||
      item.employmentStatus.toLowerCase().includes(value) ||
      item.annualIncome.toString().includes(value) || // Convert annualIncome to string and check if it includes the search value
      item.loanAmount.toString().includes(value) || // Convert loanAmount to string and check if it includes the search value
      item.loanTerm.toString().includes(value) || // Convert loanTerm to string and check if it includes the search value
      item.cibilScore.toString().includes(value) || // Convert cibilScore to string and check if it includes the search value
      (item.predictionResult && item.predictionResult.Loan_Status.toLowerCase().includes(value)) // Check if predictionResult exists and its Loan_Status includes the search value
    );
    setFilteredLoanData(filtered);
  };
  

  const handleCreditRiskSearch = (event) => {
    const value = event.target.value.toLowerCase();
    setCreditRiskSearchText(value); // Update credit risk search text

      const filtered = creditRiskUserData.filter(item =>
        item.name.toLowerCase().includes(value) ||
        item.email.toLowerCase().includes(value) ||
        item.person_age.toString().includes(value) || // Convert person_age to string and check if it includes the search value
        item.person_income.toString().includes(value) || // Convert person_income to string and check if it includes the search value
        item.person_home_ownership.toLowerCase().includes(value) ||
        item.person_emp_length.toString().includes(value) || // Convert person_emp_length to string and check if it includes the search value
        item.loan_intent.toLowerCase().includes(value) ||
        item.loan_amnt.toString().includes(value) || // Convert loan_amnt to string and check if it includes the search value
        item.loan_int_rate.toString().includes(value) || // Convert loan_int_rate to string and check if it includes the search value
        item.loan_percent_income.toString().includes(value) || // Convert loan_percent_income to string and check if it includes the search value
        item.cb_person_default_on_file.toLowerCase().includes(value) ||
        item.cb_person_cred_hist_length.toString().includes(value) || // Convert cb_person_cred_hist_length to string and check if it includes the search value
        (item.responseData && item.responseData.Prediction.toLowerCase().includes(value)) || // Check if responseData exists and its Prediction includes the search value
        (item.responseData && item.responseData['Probability of Default'].toString().includes(value)) // Check if responseData exists and its 'Probability of Default' includes the search value
      );
      setFilteredCreditRiskData(filtered);
    };
    
  
  const loanColumns = [
    {
      name: '#',
      selector: (row, index) => index + 1,
      sortable: true,
      width: '50px',
      cell: (row, index) => (filteredLoanData.indexOf(row) + 1),
    },
    {
      name: 'Name',
      selector: row => row.name,
      sortable: true,
    },
    {
      name: 'Age',
      selector: row => row.age,
      sortable: true,
    },
    {
      name: 'Email',
      selector: row => row.email,
      sortable: true,
    },
    {
      name: 'Phone',
      selector: row => row.phone,
      sortable: true,
    },
    {
      name: 'Education',
      selector: row => row.education,
      sortable: true,
    },
    {
      name: 'Employment Status',
      selector: row => row.employmentStatus,
      sortable: true,
    },
    {
      name: 'Annual Income',
      selector: row => row.annualIncome,
      sortable: true,
    },
    {
      name: 'Loan Amount',
      selector: row => row.loanAmount,
      sortable: true,
    },
    {
      name: 'Loan Term',
      selector: row => row.loanTerm,
      sortable: true,
    },
    {
      name: 'CIBIL Score',
      selector: row => row.cibilScore,
      sortable: true,
    },
    {
      name: 'Loan Status',
      selector: row => row.predictionResult.Loan_Status,
      sortable: true,
    },
    
  ];

  const creditRiskColumns = [
    {
      name: 'Name',
      selector: row => row.name,
      sortable: true,
    },
    {
      name: 'Phone',
      selector: row => row.phone,
      sortable: true,
    },
    {
      name: 'Email',
      selector: row => row.email,
      sortable: true,
    },
    {
      name: 'Age',
      selector: row => row.person_age,
      sortable: true,
    },
    {
      name: 'Income',
      selector: row => row.person_income,
      sortable: true,
    },
    {
      name: 'Home Ownership',
      selector: row => row.person_home_ownership,
      sortable: true,
    },
    {
      name: 'Employment Length',
      selector: row => row.person_emp_length,
      sortable: true,
    },
    {
      name: 'Loan Intent',
      selector: row => row.loan_intent,
      sortable: true,
    },
    {
      name: 'Loan Amount',
      selector: row => row.loan_amnt,
      sortable: true,
    },
    {
      name: 'Loan Interest Rate',
      selector: row => row.loan_int_rate,
      sortable: true,
    },
    {
      name: 'Loan Percent Income',
      selector: row => row.loan_percent_income,
      sortable: true,
    },
    {
      name: 'Default on File',
      selector: row => row.cb_person_default_on_file,
      sortable: true,
    },
    {
      name: 'Credit History Length',
      selector: row => row.cb_person_cred_hist_length,
      sortable: true,
    },
    {
      name: 'Prediction',
      selector: row => row.responseData.Prediction,
      sortable: true,
    },
    {
      name: 'Probability of Default',
      selector: row => row.responseData['Probability of Default'],
      sortable: true,
    },
    
  ];
// charts
  const fetchApprovalData = async () => {
    try {
      const response = await fetch('http://localhost:5000/calculate_approval_rate');
      const data = await response.json();
      setApprovalData(data);
      drawApprovalPieChart(data);
    } catch (error) {
      console.error('Error fetching approval data:', error);
    }
  };

  const fetchCreditRiskData = async () => {
    try {
      const response = await fetch('http://localhost:5000/calculate_creditrisk_rate');
      const data = await response.json();
      setCreditRiskData(data);
      drawCreditRiskPieChart(data);
    } catch (error) {
      console.error('Error fetching credit risk data:', error);
    }
  };

  const fetchEducationData = async () => {
    try {
      const response = await fetch('http://localhost:5000/plot_education_distribution');
      const data = await response.json();
      setEducationData(data);
      drawEducationBarChart(data);
    } catch (error) {
      console.error('Error fetching education data:', error);
    }
  };

  const fetchSelfEmployedData = async () => {
    try {
      const response = await fetch('http://localhost:5000/plot_self_employed_distribution');
      const data = await response.json();
      setSelfEmployedData(data);
      drawSelfEmployedBarChart(data);
    } catch (error) {
      console.error('Error fetching self-employed data:', error);
    }
  };



  const fetchApprovalHistogramData = async () => {
    try {
      const response = await fetch('http://localhost:5000/income_distribution_loan_approval');
      const data = await response.json();
      setApprovalHistogramData(data);
      drawApprovalHistogram(data);
    } catch (error) {
      console.error('Error fetching loan approval histogram data:', error);
    }
  };

  const fetchCreditRiskHistogramData = async () => {
    try {
      const response = await fetch('http://localhost:5000/income_distribution_credit_risk');
      const data = await response.json();
      setCreditRiskHistogramData(data);
      drawCreditRiskHistogram(data);
    } catch (error) {
      console.error('Error fetching credit risk histogram data:', error);
    }
  };

  const drawApprovalPieChart = (data) => {
    const ctx = document.getElementById('approvalPieChart');
    new Chart(ctx, {
      type: 'pie',
      data: {
        labels: [`Approved (${data['Approval Rate'].toFixed(2)}%) - ${data.Approved}`, `Rejected (${data['Rejected Rate'].toFixed(2)}%) - ${data.Rejected}`],
        datasets: [
          {
            data: [data.Approved, data.Rejected],
            backgroundColor: ['rgba(255, 99, 132, 0.8)', 'rgba(255, 205, 86, 0.8)'],
            borderColor: "White",
            borderWidth: 3,
            borderAlign: "inner"
          },
        ],
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: 'Loan Approval Status',
            position: 'top',
          },
        },
      },
    });
  };

  const drawCreditRiskPieChart = (data) => {
    const ctx = document.getElementById('creditRiskPieChart');
    new Chart(ctx, {
      type: 'pie',
      data: {
        labels: [`Default (${data['Default Rate'].toFixed(2)}%) - ${data.Default}`, `Not Default (${data['Not Default Rate'].toFixed(2)}%) - ${data['Not Default']}`],
        datasets: [
          {
            data: [data.Default, data['Not Default']],
            backgroundColor: ['rgba(255, 205, 86,0.8 )', 'rgba(54, 162, 235,0.8)'],
            borderColor: "White",
            borderWidth: 3,
            borderAlign: "inner"
          },
        ],
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: 'Credit Risk Status',
            position: 'top',
          },
        },
      },
    });
  };

  const drawEducationBarChart = (data) => {
    const ctx = document.getElementById('educationBarChart');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Not Graduate', 'Graduate'],
        datasets: [{
          label: 'Education Distribution',
          data: [data['0'], data['1']],
          backgroundColor: ['#ffcccb', '#add8e6'],
          borderWidth: 1
        }]
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: 'Education Distribution',
            position: 'top'
          }
        }
      }
    });
  };

  const drawSelfEmployedBarChart = (data) => {
    const ctx = document.getElementById('selfEmployedBarChart');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Not Self-Employed', 'Self-Employed'],
        datasets: [{
          label: 'Self-Employed Distribution',
          data: [data['0'], data['1']],
          backgroundColor: ['#C0C0C0', '#FF474C'],
          borderWidth: 1
        }]
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: 'Self-Employed Distribution',
            position: 'top'
          }
        }
      }
    });
  };

  const drawApprovalHistogram = (data) => {
    const ctx = document.getElementById('approvalHistogramChart');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.income_brackets,
        datasets: [{
          label: 'Loan Approval Income Distribution',
          data: data.customer_count,
          backgroundColor: 'rgba(160, 32, 240, 0.8)',
          borderWidth: 1,
          borderColor: "White"
        }]
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: 'Loan Approval Income Distribution',
            position: 'top'
          }
        }
      }
    });
  };

  const drawCreditRiskHistogram = (data) => {
    const ctx = document.getElementById('creditRiskHistogramChart');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.income_brackets,
        datasets: [{
          label: 'Credit Risk Income Distribution',
          data: data.customer_count,
          backgroundColor: 'rgba(186, 184, 108, 0.8)',
          borderColor: "White",
          borderWidth: 1,

        }]
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: 'Credit Risk Income Distribution',
            position: 'top'
          }
        }
      }
    });
  };

  const fetchCibilScoreData = async () => {
    try {
      const response = await fetch('http://localhost:5000/cibil_score_approval_status');
      const data = await response.json();
      setCibilScoreData(data);
      drawCibilScoreBarChart(data);
    } catch (error) {
      console.error('Error fetching CIBIL score data:', error);
    }
  };

  const drawCibilScoreBarChart = (data) => {
    const ctx = document.getElementById('cibilScoreBarChart');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: Object.keys(data),
        datasets: [
          {
            label: 'Approved',
            data: Object.values(data).map(item => item.Approved),
            backgroundColor: 'rgba(0, 0, 139, 0.8)',
          },
          {
            label: 'Rejected',
            data: Object.values(data).map(item => item.Rejected),
            backgroundColor: 'rgba(255, 255, 0, 0.8)',
          },
        ],
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: 'Loan Approval Status by CIBIL Score Range',
            position: 'top',
          },
        },
        scales: {
          x: {
            stacked: true,
            title: {
              display: true,
              text: 'CIBIL Score Range',
            },
          },
          y: {
            stacked: true,
            title: {
              display: true,
              text: 'Number of Customers',
            },
          },
        },
      },
    });
  };

  const fetchAgeDistributionData = async () => { // Added
    try {
      const response = await fetch('http://localhost:5000/age_distribution_credit_risk'); // Endpoint for age distribution
      const data = await response.json();
      setAgeDistributionData(data);
      drawAgeDistributionDonutChart(data); // Added
    } catch (error) {
      console.error('Error fetching age distribution data:', error);
    }
  };

  const drawAgeDistributionDonutChart = (data) => { // Added
    const ctx = document.getElementById('ageDistributionDonutChart');
    new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: data.age_groups,
        datasets: [
          {
            label: 'Age Distribution',
            data: data.customer_count,
            backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#8A2BE2', '#32CD32', '#000000'],
            borderWidth: 2,
            borderColor: "White",
            borderAlign: "inner"
          }
        ]
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: 'Age Distribution for Credit Risk',
            position: 'top'
          }
        }
      }
    });
  };

  return (
    <>
      <Header />
      <SideMenu />
      <main id="main" className="main">
        <Breadcrumb />
        <section className="section dashboard">
          <div className="row">
          <div className="col-lg-4">
              <div className="card">
                <div className="card-body">
                  <h5 className="card-title">Loan Approval Status</h5>
                  <canvas id="approvalPieChart" />
                </div>
              </div>
            </div>
            <div className="col-lg-4">
              <div className="card">
                <div className="card-body">
                  <h5 className="card-title">Credit Risk Status</h5>
                  <canvas id="creditRiskPieChart" />
                </div>
              </div>
            </div>
          </div>
          <div className="row mt-4">
            <div className="col-lg-6">
              <div className="card">
                <div className="card-body">
                  <h5 className="card-title">Education Distribution</h5>
                  <canvas id="educationBarChart" />
                </div>
              </div>
            </div>
            <div className="col-lg-6">
              <div className="card">
                <div className="card-body">
                  <h5 className="card-title">Self-Employed Distribution</h5>
                  <canvas id="selfEmployedBarChart" />
                </div>
              </div>
            </div>
          </div>
          <div className="row mt-4">
            <div className="col-lg-6">
              <div className="card">
                <div className="card-body">
                  <h5 className="card-title">Loan Approval Income Distribution</h5>
                  <canvas id="approvalHistogramChart" />
                </div>
              </div>
            </div>
            <div className="col-lg-6">
              <div className="card">
                <div className="card-body">
                  <h5 className="card-title">Credit Risk Income Distribution</h5>
                  <canvas id="creditRiskHistogramChart" />
                </div>
              </div>
            </div>
          </div>
          <div className="row mt-4">
            <div className="col-lg-8">
              <div className="card">
                <div className="card-body">
                  <h5 className="card-title">Loan Approval Status by CIBIL Score Range</h5>
                  <canvas id="cibilScoreBarChart" />
                </div>
              </div>
            </div>
          </div>
          <div className="col-lg-4">
            <div className="card">
              <div className="card-body">
                <h5 className="card-title">Age Distribution for Credit Risk</h5> {/* Added */}
                <canvas id="ageDistributionDonutChart" />
              </div>
            </div>
          </div>
          <div className="row">
            {/* Loan Approval Data */}
            <div className="col-lg-12">
              <div className="row">
                <div className="col-12">
                  <div className="card recent-sales overflow-auto">
                    <div className="card-body">
                      <h5 className="card-title">Loan Approval Data </h5>
                      <input
                        type="text"
                        placeholder="Search Loan Data"
                        value={loanSearchText}
                        onChange={handleLoanSearch}
                        className="form-control mb-3"
                      />
                      <DataTable
                        columns={loanColumns}
                        data={filteredLoanData}
                        pagination
                        paginationPerPage={10}
                        highlightOnHover
                        pointerOnHover
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Credit Risk Data */}
            <div className="col-lg-12">
              <div className="row">
                <div className="col-12">
                  <div className="card top-selling overflow-auto">
                    <div className="card-body">
                      <h5 className="card-title">Credit Risk Data </h5>
                      <input
                        type="text"
                        placeholder="Search Credit Risk Data"
                        value={creditRiskSearchText}
                        onChange={handleCreditRiskSearch}
                        className="form-control mb-3"
                      />
                      <DataTable
                        columns={creditRiskColumns}
                        data={filteredCreditRiskData}
                        pagination
                        paginationPerPage={10}
                        highlightOnHover
                        pointerOnHover
                      />
                      
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </>
  );
};

export default Dashboard;
