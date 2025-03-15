import React, { useState } from 'react';
import { 
  Container, Typography, Paper, Button, Box, 
  CircularProgress 
} from '@mui/material';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

const Login = () => {
  const { googleLogin } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleGoogleLogin = async () => {
    try {
      setError('');
      setLoading(true);
      await googleLogin();
      navigate('/dashboard');
    } catch (err) {
      setError('Failed to log in with Google. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="xs">
      <Paper 
        elevation={4}
        sx={{
          padding: '2rem',
          marginTop: '5rem',
          borderRadius: '12px',
          boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.1)'
        }}
      >
        <Box display="flex" flexDirection="column" alignItems="center">
          {/* Logo or Heading */}
          <Typography 
            variant="h4" 
            gutterBottom
            sx={{
              fontWeight: '600',
              color: '#333',
              textAlign: 'center'
            }}
          >
            Welcome to Carbon Bytes
          </Typography>

          {/* Subheading */}
          <Typography 
            variant="body1" 
            align="center"
            sx={{
              color: '#666',
              marginBottom: '1rem',
              fontSize: '1rem'
            }}
          >
            Join our community to help reduce waste and earn rewards for your environmental contributions.
          </Typography>

          {/* Error Message */}
          {error && (
            <Typography 
              variant="body2" 
              color="error" 
              sx={{ marginBottom: '1rem' }}
            >
              {error}
            </Typography>
          )}

          {/* Google Login Button */}
          <Button
            variant="contained"
            fullWidth
            onClick={handleGoogleLogin}
            disabled={loading}
            sx={{
              backgroundColor: '#4285F4',
              color: '#fff',
              padding: '0.6rem',
              fontSize: '0.9rem',
              fontWeight: '500',
              borderRadius: '8px',
              textTransform: 'none',
              '&:hover': {
                backgroundColor: '#357ae8'
              },
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              gap: '0.5rem',
              mt: 2
            }}
          >
            {loading ? (
              <CircularProgress size={24} color="inherit" />
            ) : (
              <>
                <img alt="svgImg" src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIj8+PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciICB2aWV3Qm94PSIwIDAgMzAgMzAiIHdpZHRoPSIzMHB4IiBoZWlnaHQ9IjMwcHgiPiAgICA8cGF0aCBkPSJNIDE1LjAwMzkwNiAzIEMgOC4zNzQ5MDYyIDMgMyA4LjM3MyAzIDE1IEMgMyAyMS42MjcgOC4zNzQ5MDYyIDI3IDE1LjAwMzkwNiAyNyBDIDI1LjAxMzkwNiAyNyAyNy4yNjkwNzggMTcuNzA3IDI2LjMzMDA3OCAxMyBMIDI1IDEzIEwgMjIuNzMyNDIyIDEzIEwgMTUgMTMgTCAxNSAxNyBMIDIyLjczODI4MSAxNyBDIDIxLjg0ODcwMiAyMC40NDgyNTEgMTguNzI1OTU1IDIzIDE1IDIzIEMgMTAuNTgyIDIzIDcgMTkuNDE4IDcgMTUgQyA3IDEwLjU4MiAxMC41ODIgNyAxNSA3IEMgMTcuMDA5IDcgMTguODM5MTQxIDcuNzQ1NzUgMjAuMjQ0MTQxIDguOTY4NzUgTCAyMy4wODU5MzggNi4xMjg5MDYyIEMgMjAuOTUxOTM3IDQuMTg0OTA2MyAxOC4xMTY5MDYgMyAxNS4wMDM5MDYgMyB6Ii8+PC9zdmc+"/>
                Sign in with Google
              </>
            )}
          </Button>
        </Box>
      </Paper>
    </Container>
  );
};

export default Login;
