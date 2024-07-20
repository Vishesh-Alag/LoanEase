import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Login from '../pages/admin/Login';
import Register from '../pages/admin/Register';

  import "../assets/vendor/bootstrap/css/bootstrap.min.css"
  import "../assets/vendor/bootstrap-icons/bootstrap-icons.css"
  import "../assets/vendor/boxicons/css/boxicons.min.css"
  import "../assets/vendor/quill/quill.snow.css"
  import "../assets/vendor/quill/quill.bubble.css"
  import "../assets/vendor/remixicon/remixicon.css"
  import "../assets/vendor/simple-datatables/style.css"
  import "../assets/css/style.css"
  // import "../assets/vendor/apexcharts/apexcharts.min.js"
  import "../assets/vendor/bootstrap/js/bootstrap.bundle.min.js"
  import "../assets/vendor/chart.js/chart.umd.js"
  import "../assets/vendor/echarts/echarts.min.js"
  import "../assets/vendor/quill/quill.min.js"
  import "../assets/vendor/simple-datatables/simple-datatables.js"
  import "../assets/vendor/tinymce/tinymce.min.js"
  import "../assets/vendor/php-email-form/validate.js"
 
  import "../assets/js/main.js"
import Dashboard from '../pages/admin/Dashboard.js';
import Error404 from '../pages/Error404.js';
import BankRegister from '../pages/admin/BankRegister.js';
import Profile from '../pages/admin/Profile.js';

  
  
const Adminroutes = () => {
  return (
  <Routes> 
     <Route exact path='/login' element={<Login />} />
     <Route exact path='/' element={<Login />} />
     <Route exact path='/bank-register' element={<BankRegister />} />
     <Route exact path='/dashboard' element={<Dashboard />} />
     <Route exact path='/register' element={<Register />} />
     <Route exact path='/profile' element={<Profile />} />
     <Route exact path='/*' element={<Error404/>} />
  </Routes>
)
}

export default Adminroutes