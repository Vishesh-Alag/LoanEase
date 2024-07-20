import React, { useState, useEffect } from 'react';
import Header from '../../components/admin/Header';
import Breadcrumb from '../../components/admin/Breadcrumb';
import SideMenu from '../../components/admin/SideMenu';
import Footer from '../../components/admin/Footer';
import { useCookies } from 'react-cookie';
import DataTable from 'react-data-table-component';
import Modal from 'react-modal';

const Dashboard = () => {
  const [cookies] = useCookies(['loggedIn']);
  const [bankData, setBankData] = useState([]);
  const [searchText, setSearchText] = useState('');
  const [filteredBankData, setFilteredBankData] = useState([]);
  const [queries, setQueries] = useState([]);
  const [filteredQueryData, setFilteredQueryData] = useState([]);
  const [modalIsOpen, setModalIsOpen] = useState(false);
  const [selectedQuery, setSelectedQuery] = useState({});
  const [response, setResponse] = useState('');
  useEffect(() => {
    // Fetch bank data from the API
    fetch('http://localhost:5000/get_all_banks')
      .then(response => response.json())
      .then(data => {
        setBankData(data);
        setFilteredBankData(data);
      })
      .catch(error => console.error('Error fetching bank data:', error));

    // Fetch queries data from the API
    fetch('http://localhost:5000/get_queries')
      .then(response => response.json())
      .then(data => {
        setQueries(data);
        setFilteredQueryData(data);
      })
      .catch(error => console.error('Error fetching queries:', error));
  }, []);

  // If the loggedIn cookie is not set, redirect to the login page
  useEffect(() => {
    if (!cookies.adminLoggedIn) {
      window.location.href = '/admin/login';
    }
  }, [cookies.adminLoggedIn]);

  const handleBankSearch = (value) => {
    setSearchText(value);

    const filtered = bankData.filter(item =>
      item.bankName.toLowerCase().includes(value.toLowerCase()) ||
      item.branchAddress.toLowerCase().includes(value.toLowerCase()) ||
      item.ifscCode.toLowerCase().includes(value.toLowerCase()) ||
      item.primaryContactName.toLowerCase().includes(value.toLowerCase()) ||
      item.primaryContactEmail.toLowerCase().includes(value.toLowerCase()) ||
      item.primaryContactPhone.toLowerCase().includes(value.toLowerCase())
    );
    setFilteredBankData(filtered);
  };

  const handleQuerySearch = (value) => {
    setSearchText(value);

    const filtered = queries.filter(item =>
      item.name.toLowerCase().includes(value.toLowerCase()) ||
      item.subject.toLowerCase().includes(value.toLowerCase()) ||
      item.date.toLowerCase().includes(value.toLowerCase()) ||
      item.timestamp.toLowerCase().includes(value.toLowerCase())
    );
    setFilteredQueryData(filtered);
  };

  const handleViewDetails = (query) => {
    setSelectedQuery(query);
    fetchQueryDetails(query.id); // Fetch query details based on its ID
    setModalIsOpen(true);
  };

  const fetchQueryDetails = (queryId) => {
    // Fetch details of the selected query using its ID
    fetch(`http://localhost:5000/get_query_details/${queryId}`)
      .then(response => response.json())
      .then(data => {
        setResponse(data.message); // Assuming the API returns a 'response' field
      })
      .catch(error => console.error('Error fetching query details:', error));
  };
  const handleResponse = () => {
    // Assuming the API returns receiver email in the query data
    const receiverEmail = selectedQuery.email;

    // Open Gmail compose URL in a new tab
    window.open(`https://mail.google.com/mail/?view=cm&fs=1&to=${receiverEmail}&su=Response%20to%20Query&body=${encodeURIComponent(response)}`);

    // Close the modal
    closeModal();
  };
  const closeModal = () => {
    setModalIsOpen(false);
  };

  const bank_columns = [
    {
      name: '#',
      selector: (row, index) => index + 1,
      sortable: true,
    },
    {
      name: 'Bank Name',
      selector: row => row.bankName,
      sortable: true,
    },
    {
      name: 'Branch Address',
      selector: row => row.branchAddress,
      sortable: true,
    },
    {
      name: 'IFSC Code',
      selector: row => row.ifscCode,
      sortable: true,
    },
    {
      name: 'Primary Contact Name',
      selector: row => row.primaryContactName,
      sortable: true,
    },
    {
      name: 'Primary Contact Email',
      selector: row => row.primaryContactEmail,
      sortable: true,
    },
    {
      name: 'Primary Contact Phone',
      selector: row => row.primaryContactPhone,
      sortable: true,
    },
  ];

  const query_columns = [
    {
      name: '#',
      selector: (row, index) => index + 1,
      sortable: true,
    },
    {
      name: 'Name',
      selector: row => row.name,
      sortable: true,
    },
    {
      name: 'Subject',
      selector: row => row.subject,
      sortable: true,
    },
    {
      name: 'Date',
      selector: row => row.date,
      sortable: true,
    },
    {
      name: 'Time',
      selector: row => row.timestamp,
      sortable: true,
    },
    // Query columns definition
    {
      name: 'Action',
      cell: row => (
        <button className="btn btn-primary" onClick={() => handleViewDetails(row)}>View</button>
      ),
      ignoreRowClick: true,
      allowOverflow: true,
      button: true,
    },
  ];

  const bankSearchProps = {
    placeholder: 'Search',
    value: searchText,
    onChange: (e) => handleBankSearch(e.target.value),
    className: 'form-control mb-3',
  };

  const querySearchProps = {
    placeholder: 'Search',
    value: searchText,
    onChange: (e) => handleQuerySearch(e.target.value),
    className: 'form-control mb-3',
  };

  return (
    <>
      <Header />
      <SideMenu />
      <main id="main" className="main">
      <Breadcrumb active="Dashboard" />
        <section className="section dashboard">
          <div className="row">
            <div className="col-lg-12">
              <div className="row">
                <div className="col-12">
                  <div className="card recent-sales overflow-auto">
                    <div className="card-body">
                      <h5 className="card-title">Banks Display </h5>
                      <input {...bankSearchProps} />
                      <DataTable columns={bank_columns} data={filteredBankData} pagination={true} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
        <section className="section dashboard">
          <div className="row">
            <div className="col-lg-12">
              <div className="row">
                <div className="col-12">
                  <div className="card recent-sales overflow-auto">
                    <div className="card-body">
                      <h5 className="card-title">Queries Display </h5>
                      <input {...querySearchProps} />
                      <DataTable columns={query_columns} data={filteredQueryData} pagination={true} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

        </section>
      </main>
      <Footer />
      <Modal
        isOpen={modalIsOpen}
        onRequestClose={closeModal}
        className="modal-dialog modal-dialog-centered"
        style={{ maxWidth: '400px', margin: 'auto' }} // Decreased width and added margin
      >
        <div className="modal-content">
          <div className="modal-header">
            <h5 className="modal-title">Query Details</h5>
            <button type="button" className="close" onClick={closeModal}>
              <span>&times;</span>
            </button>
          </div>
          <div className="modal-body" style={{ marginTop: '50px', marginLeft: '400px', marginRight: '400px' }}> {/* Added margin top and adjusted horizontal margin */}
            <div className="card" style={{ width: '100%' }}> {/* Set width to 100% to fill modal body */}
              <div className="card-body" > {/* Added margin auto to center content within card */}
                <h5 className="card-title">Query Details</h5>
                <p><strong>Name:</strong> {selectedQuery.name}</p>
                <p><strong>Email:</strong> {selectedQuery.email}</p>
                <p><strong>Subject:</strong> {selectedQuery.subject}</p>
                <p><strong>Date:</strong> {selectedQuery.date}</p>
                <p><strong>Time:</strong> {selectedQuery.timestamp}</p>
                <p><strong>Message:</strong> {selectedQuery.message}</p>
                <div className="row">
                  <div className="col">
                    <button type="button" className="btn btn-primary mr-2" onClick={handleResponse}>Respond</button>
                  </div>
                  <div className="col">
                    <button type="button" className="btn btn-secondary" onClick={closeModal}>Close</button>
                  </div>
                </div>
              </div>
            </div>
          </div>
      </div>
      </Modal>



    </>
  );
};

export default Dashboard;
