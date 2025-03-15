import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box, Avatar } from '@mui/material';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

const Header = () => {
  const { currentUser, logout } = useAuth();

  return (
    <AppBar position="static" style={{ backgroundColor: '#2e7d32' }}>
      <Toolbar>
        <Typography variant="h6" style={{ flexGrow: 1 }}>
          <Link to="/" style={{ color: 'white', textDecoration: 'none' }}>
            Carbon Bytes
          </Link>
        </Typography>
        <Box>
          <Button color="inherit" component={Link} to="/">
            Home
          </Button>
          
          {currentUser ? (
            <>
              <Button color="inherit" component={Link} to="/dashboard">
                Dashboard
              </Button>
              <Button color="inherit" component={Link} to="/scan">
                Scan QR
              </Button>
              <Button color="inherit" component={Link} to="/leaderboard">
                Leaderboard
              </Button>
              <Button 
                color="inherit" 
                component={Link} 
                to="/profile"
                style={{ marginRight: '8px' }}
              >
                Profile
              </Button>
              <Button color="inherit" onClick={logout}>
                Logout
              </Button>
              <Link to="/profile">
                <Avatar 
                  src={currentUser.photoURL} 
                  alt={currentUser.displayName}
                  style={{ width: '32px', height: '32px', marginLeft: '8px', cursor: 'pointer' }}
                />
              </Link>
            </>
          ) : (
            <Button color="inherit" component={Link} to="/login">
              Login
            </Button>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;