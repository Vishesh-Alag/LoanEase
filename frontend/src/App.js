import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Adminroutes from './routes/Adminroutes';
import Bankroutes from './routes/Bankroutes';
import Error404 from './pages/Error404';
const App = () => {


  return (
    <Router>
     
      <Routes>
        <Route exact path='/admin/*' element={<Adminroutes />} />
        <Route exact path='/*' element={<Error404 />} />
        <Route exact path='/bank/*' element={<Bankroutes />} />
        <Route exact path='/' element={<Bankroutes />} />
      </Routes>
    </Router >
  );
};

export default App;