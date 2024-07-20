import React from 'react'
import Header from '../../components/bank/Header'
import Breadcrumb from '../../components/bank/Breadcrumb'
import SideMenu from '../../components/bank/SideMenu'
import Footer from '../../components/bank/Footer'

const FAQ = () => {
  return (
    <>
      <Header />
      <SideMenu />
      <main id="main" className="main">
        <Breadcrumb />
        <section className="section faq">
          <div className="row">
            <div className="col-lg-6">

              {/* F.A.Q Group 1 */}
              <div className="card">
                <div className="card-body">
                  <h5 className="card-title">F.A.Q's</h5>
                  <div className="accordion accordion-flush" id="faq-group-1">
                    <div className="accordion-item">
                      <h2 className="accordion-header">
                        <button className="accordion-button collapsed" data-bs-target="#faqsOne-1" type="button" data-bs-toggle="collapse">
                          What is the purpose of this web app?
                        </button>
                      </h2>
                      <div id="faqsOne-1" className="accordion-collapse collapse" data-bs-parent="#faq-group-1">
                        <div className="accordion-body">
                          This web app aims to simplify the loan eligibility process for banks, reduce manpower, and provide some automation in the loan approval and credit risk assessment processes.                  </div>
                      </div>
                    </div>
                    <div className="accordion-item">
                      <h2 className="accordion-header">
                        <button className="accordion-button collapsed" data-bs-target="#faqsOne-2" type="button" data-bs-toggle="collapse">
                          How does the registration process work for banks and their branches?
                        </button>
                      </h2>
                      <div id="faqsOne-2" className="accordion-collapse collapse" data-bs-parent="#faq-group-1">
                        <div className="accordion-body">
                          Bank branches can register on the platform by providing their details such as branch name, location, and contact information. Each branch will have its own login credentials to access the service.                  </div>
                      </div>
                    </div>
                    <div className="accordion-item">
                      <h2 className="accordion-header">
                        <button className="accordion-button collapsed" data-bs-target="#faqsOne-3" type="button" data-bs-toggle="collapse">
                          Can multiple employees from the same bank branch access the platform with the same credentials?
                        </button>
                      </h2>
                      <div id="faqsOne-3" className="accordion-collapse collapse" data-bs-parent="#faq-group-1">
                        <div className="accordion-body">
                          Yes, multiple employees from the same bank branch can access the platform using the branch's login credentials.                  </div>
                      </div>
                    </div>
                    <div className="accordion-item">
                      <h2 className="accordion-header">
                        <button className="accordion-button collapsed" data-bs-target="#faqsOne-4" type="button" data-bs-toggle="collapse">
                          Who has access to the employee details stored in the dashboard?
                        </button>
                      </h2>
                      <div id="faqsOne-4" className="accordion-collapse collapse" data-bs-parent="#faq-group-1">
                        <div className="accordion-body">
                          Only the developer who develops this web app has access to the dashboard where employee details are stored securely.                  </div>
                      </div>
                    </div>
                    <div className="accordion-item">
                      <h2 className="accordion-header">
                        <button className="accordion-button collapsed" data-bs-target="#faqsOne-5" type="button" data-bs-toggle="collapse">
                          What information is required to determine a customer's loan eligibility?
                        </button>
                      </h2>
                      <div id="faqsOne-5" className="accordion-collapse collapse" data-bs-parent="#faq-group-1">
                        <div className="accordion-body">
                          Basic details such as name, age, gender, contact information, income, employment status, loan amount, loan term, credit score, and asset values are required.                  </div>
                      </div>
                    </div>
                    <div className="accordion-item">
                      <h2 className="accordion-header">
                        <button className="accordion-button collapsed" data-bs-target="#faqsOne-6" type="button" data-bs-toggle="collapse">
                          How is the loan eligibility status determined?                        </button>
                      </h2>
                      <div id="faqsOne-6" className="accordion-collapse collapse" data-bs-parent="#faq-group-1">
                        <div className="accordion-body">
                          The loan eligibility status is calculated using a machine learning model integrated into the backend of the platform. This model analyzes the customer's details and generates a report indicating whether they are eligible for a loan or not.                      </div>
                      </div>
                    </div>
                  </div>
                </div>{/* End F.A.Q Group 1 */}
              </div>
            </div>
            <div className="col-lg-6">
              {/* F.A.Q Group 2 */}
              <div className="card">
                <div className="card-body">
                  <h5 className="card-title">More F.A.Q's</h5>
                  <div className="accordion accordion-flush" id="faq-group-2">
                    <div className="accordion-item">
                      <h2 className="accordion-header">
                        <button className="accordion-button collapsed" data-bs-target="#faqsTwo-1" type="button" data-bs-toggle="collapse">
                          What happens after the loan eligibility status is determined?            </button>
                      </h2>
                      <div id="faqsTwo-1" className="accordion-collapse collapse" data-bs-parent="#faq-group-2">
                        <div className="accordion-body">
                          A report is generated with the loan eligibility status and basic analysis, which is displayed on the UI. Additionally, an email is sent from the bank branch account to the customer, informing them of the loan status.            </div>
                      </div>
                    </div>
                    <div className="accordion-item">
                      <h2 className="accordion-header">
                        <button className="accordion-button collapsed" data-bs-target="#faqsTwo-2" type="button" data-bs-toggle="collapse">
                          What is the purpose of the Credit Risk Assessment service?            </button>
                      </h2>
                      <div id="faqsTwo-2" className="accordion-collapse collapse" data-bs-parent="#faq-group-2">
                        <div className="accordion-body">
                          The Credit Risk Assessment service predicts whether a customer is likely to default on a loan and calculates the percentage of default risk. This helps banks make informed decisions about lending.            </div>
                      </div>
                    </div>
                    <div className="accordion-item">
                      <h2 className="accordion-header">
                        <button className="accordion-button collapsed" data-bs-target="#faqsTwo-3" type="button" data-bs-toggle="collapse">
                          What information is required for Credit Risk Assessment?            </button>
                      </h2>
                      <div id="faqsTwo-3" className="accordion-collapse collapse" data-bs-parent="#faq-group-2">
                        <div className="accordion-body">
                          Similar to loan eligibility, basic customer details such as name, age, contact information, income, loan amount, loan intent, and credit history are required.            </div>
                      </div>
                    </div>
                    <div className="accordion-item">
                      <h2 className="accordion-header">
                        <button className="accordion-button collapsed" data-bs-target="#faqsTwo-4" type="button" data-bs-toggle="collapse">
                          How is the Credit Risk Assessment calculated?            </button>
                      </h2>
                      <div id="faqsTwo-4" className="accordion-collapse collapse" data-bs-parent="#faq-group-2">
                        <div className="accordion-body">
                          The Credit Risk Assessment is also performed using a machine learning model integrated into the backend. The model analyzes customer details and generates a report indicating the likelihood of default and the percentage of default risk.            </div>
                      </div>
                    </div>
                    <div className="accordion-item">
                      <h2 className="accordion-header">
                        <button className="accordion-button collapsed" data-bs-target="#faqsTwo-5" type="button" data-bs-toggle="collapse">
                          Can customers access this platform directly?            </button>
                      </h2>
                      <div id="faqsTwo-5" className="accordion-collapse collapse" data-bs-parent="#faq-group-2">
                        <div className="accordion-body">
                          No, this platform is designed for bank employees to streamline the loan approval and credit risk assessment processes. Customers interact with their bank directly for loan applications and inquiries.            </div>
                      </div>
                    </div>
                    <div className="accordion-item">
                      <h2 className="accordion-header">
                        <button className="accordion-button collapsed" data-bs-target="#faqsTwo-6" type="button" data-bs-toggle="collapse">
                          How does the platform handle errors or discrepancies in customer data?            </button>
                      </h2>
                      <div id="faqsTwo-6" className="accordion-collapse collapse" data-bs-parent="#faq-group-2">
                        <div className="accordion-body">
                          Our platform includes validation mechanisms to identify and flag errors or discrepancies in customer data inputted by bank employees. This helps ensure the accuracy and reliability of the analysis and recommendations provided.            </div>
                      </div>
                    </div>

                  </div>
                </div>

              </div>{/* End F.A.Q Group 2 */}
            </div>

          </div>
        </section>
      </main>{/* End #main */}

      <Footer />
    </>
  )
}

export default FAQ;