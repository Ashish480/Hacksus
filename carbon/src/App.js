import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import Header from './components/Header';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import Login from './pages/Login';
import QRScanner from './components/QRScanner';
import Leaderboard from './pages/Leaderboard';
import Profile from './pages/Profile';
import { createTheme, ThemeProvider } from '@mui/material/styles';

// Create a nature-friendly theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#2e7d32', // Dark green
      light: '#60ad5e',
      dark: '#005005',
    },
    secondary: {
      main: '#1b5e20', // Darker green
      light: '#4c8c4a',
      dark: '#003300',
    },
    background: {
      default: '#f5f8f5', // Light green tint
    },
  },
  typography: {
    fontFamily: "'Poppins', 'Roboto', 'Arial', sans-serif",
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 500,
    },
    h6: {
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 8,
  },
  overrides: {
    MuiButton: {
      root: {
        textTransform: 'none',
        borderRadius: 8,
      },
      contained: {
        boxShadow: 'none',
        '&:hover': {
          boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.1)',
        },
      },
    },
    MuiCard: {
      root: {
        overflow: 'hidden',
        boxShadow: '0px 2px 8px rgba(0, 0, 0, 0.05)',
      },
    },
    MuiPaper: {
      rounded: {
        borderRadius: 8,
      },
    },
  },
});

// Private route component
const PrivateRoute = ({ children }) => {
  const { currentUser } = useAuth();
  return currentUser ? children : <Navigate to="/login" />;
};

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Router>
        <AuthProvider>
          <div style={{ minHeight: '100vh', backgroundColor: '#f5f8f5' }}>
            <Header />
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/login" element={<Login />} />
              <Route 
                path="/dashboard" 
                element={
                  <PrivateRoute>
                    <Dashboard />
                  </PrivateRoute>
                } 
              />
              <Route 
                path="/scan" 
                element={
                  <PrivateRoute>
                    <QRScanner />
                  </PrivateRoute>
                } 
              />
              <Route 
                path="/leaderboard" 
                element={
                  <PrivateRoute>
                    <Leaderboard />
                  </PrivateRoute>
                } 
              />
              <Route 
                path="/profile" 
                element={
                  <PrivateRoute>
                    <Profile />
                  </PrivateRoute>
                } 
              />
            </Routes>
            <footer style={{ 
              textAlign: 'center', 
              padding: '20px', 
              marginTop: '40px',
              borderTop: '1px solid #e0e0e0',
              backgroundColor: '#f0f4f0'
            }}>
              <p>© {new Date().getFullYear()} Carbon Bytes. All rights reserved.</p>
              <p style={{ fontSize: '0.9rem', color: '#666' }}>Making our planet greener, one byte at a time.</p>
            </footer>
          </div>
        </AuthProvider>
      </Router>
    </ThemeProvider>
  );
}

export default App;