import React from 'react';
import { 
  Container, Typography, Button, Grid, Box, 
  Card, CardContent, CardMedia, Paper
} from '@mui/material';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

const Home = () => {
  const { currentUser } = useAuth();

  const features = [
    {
      title: 'Scan & Recycle',
      description: 'Scan QR codes on our smart bins to log your waste disposal and earn points.',
      icon: '♻️',
    },
    {
      title: 'Track Progress',
      description: 'Monitor your recycling habits and see your positive environmental impact.',
      icon: '📊',
    },
    {
      title: 'Earn Rewards',
      description: 'Convert your recycling points into real-world rewards and discounts.',
      icon: '🎁',
    },
    {
      title: 'Compete & Share',
      description: 'Join leaderboards and challenge friends to make a bigger impact together.',
      icon: '🏆',
    },
  ];

  const stats = [
    { value: '10,000+', label: 'Items Recycled' },
    { value: '500+', label: 'Active Users' },
    { value: '5 tons', label: 'CO₂ Saved' },
    { value: '30+', label: 'Smart Bins Deployed' },
  ];

  return (
    <>
      {/* Hero Section */}
      <Box 
        style={{ 
          background: 'linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%)',
          color: 'white',
          padding: '80px 0',
          marginBottom: '40px'
        }}
      >
        <Container>
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={7}>
              <Typography variant="h3" gutterBottom>
                Turn Your Waste Into Rewards
              </Typography>
              <Typography variant="h6" paragraph style={{ marginBottom: '30px', opacity: 0.9 }}>
                Carbon Bytes helps you track, manage, and get rewarded for your 
                recycling efforts while making a positive impact on the environment.
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={2}>
                {currentUser ? (
                  <Button 
                    variant="contained" 
                    size="large"
                    component={Link}
                    to="/dashboard"
                    style={{ backgroundColor: 'white', color: '#2e7d32', fontWeight: 'bold' }}
                  >
                    Go to Dashboard
                  </Button>
                ) : (
                  <>
                    <Button 
                      variant="contained" 
                      size="large"
                      component={Link}
                      to="/login"
                      style={{ backgroundColor: 'white', color: '#2e7d32', fontWeight: 'bold' }}
                    >
                      Get Started
                    </Button>
                    <Button 
                      variant="outlined" 
                      size="large"
                      href="#how-it-works"
                      style={{ borderColor: 'white', color: 'white' }}
                    >
                      Learn More
                    </Button>
                  </>
                )}
              </Box>
            </Grid>
            <Grid item xs={12} md={5}>
              <Box 
                style={{ 
                  backgroundColor: 'rgba(255,255,255,0.1)', 
                  borderRadius: '12px',
                  padding: '20px',
                  textAlign: 'center'
                }}
              >
                <Typography variant="h4" gutterBottom>
                  Environmental Impact
                </Typography>
                <Grid container spacing={2}>
                  {stats.map((stat, index) => (
                    <Grid item xs={6} key={index}>
                      <Box p={2}>
                        <Typography variant="h4" style={{ fontWeight: 'bold' }}>
                          {stat.value}
                        </Typography>
                        <Typography variant="body2">
                          {stat.label}
                        </Typography>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* How It Works Section */}
      <Container id="how-it-works" style={{ marginBottom: '60px' }}>
        <Box textAlign="center" mb={6}>
          <Typography variant="h4" gutterBottom>
            How Carbon Bytes Works
          </Typography>
          <Typography variant="body1" style={{ maxWidth: '700px', margin: '0 auto' }}>
            Our platform makes recycling rewarding, engaging, and easy to track.
          </Typography>
        </Box>

        <Grid container spacing={4}>
          <Grid item xs={12} md={3}>
            <Paper elevation={2} style={{ height: '100%', borderRadius: '12px', overflow: 'hidden' }}>
              <Box p={3} textAlign="center">
                <Typography variant="h1" style={{ color: '#2e7d32', marginBottom: '10px' }}>1</Typography>
                <Typography variant="h6" gutterBottom>Find a Smart Bin</Typography>
                <Typography variant="body2">
                  Locate one of our Carbon Bytes smart waste bins in your area.
                </Typography>
              </Box>
            </Paper>
          </Grid>
          <Grid item xs={12} md={3}>
            <Paper elevation={2} style={{ height: '100%', borderRadius: '12px', overflow: 'hidden' }}>
              <Box p={3} textAlign="center">
                <Typography variant="h1" style={{ color: '#2e7d32', marginBottom: '10px' }}>2</Typography>
                <Typography variant="h6" gutterBottom>Scan QR Code</Typography>
                <Typography variant="body2">
                  Use our app to scan the QR code on the bin to connect your account.
                </Typography>
              </Box>
            </Paper>
          </Grid>
          <Grid item xs={12} md={3}>
            <Paper elevation={2} style={{ height: '100%', borderRadius: '12px', overflow: 'hidden' }}>
              <Box p={3} textAlign="center">
                <Typography variant="h1" style={{ color: '#2e7d32', marginBottom: '10px' }}>3</Typography>
                <Typography variant="h6" gutterBottom>Dispose Properly</Typography>
                <Typography variant="body2">
                  Dispose your waste in the bin. Our system detects and categorizes the waste type.
                </Typography>
              </Box>
            </Paper>
          </Grid>
          <Grid item xs={12} md={3}>
            <Paper elevation={2} style={{ height: '100%', borderRadius: '12px', overflow: 'hidden' }}>
              <Box p={3} textAlign="center">
                <Typography variant="h1" style={{ color: '#2e7d32', marginBottom: '10px' }}>4</Typography>
                <Typography variant="h6" gutterBottom>Earn Rewards</Typography>
                <Typography variant="body2">
                  Get points based on waste type that can be redeemed for rewards.
                </Typography>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </Container>

      {/* Features Section */}
      <Box style={{ backgroundColor: '#f0f4f0', padding: '60px 0' }}>
        <Container>
          <Box textAlign="center" mb={6}>
            <Typography variant="h4" gutterBottom>
              Features & Benefits
            </Typography>
            <Typography variant="body1" style={{ maxWidth: '700px', margin: '0 auto' }}>
              Carbon Bytes offers a range of features to make recycling more engaging and rewarding.
            </Typography>
          </Box>

          <Grid container spacing={4}>
            {features.map((feature, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Card style={{ height: '100%', textAlign: 'center', border: 'none', boxShadow: 'none', backgroundColor: 'transparent' }}>
                  <CardContent>
                    <Typography variant="h2" style={{ marginBottom: '10px' }}>
                      {feature.icon}
                    </Typography>
                    <Typography variant="h6" gutterBottom>
                      {feature.title}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {feature.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* Call to Action */}
      <Container style={{ marginTop: '60px', marginBottom: '60px' }}>
        <Paper elevation={3} style={{ 
          borderRadius: '12px', 
          padding: '40px', 
          textAlign: 'center',
          background: 'linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%)'
        }}>
          <Typography variant="h4" gutterBottom>
            Join the Carbon Bytes Community Today
          </Typography>
          <Typography variant="body1" paragraph style={{ maxWidth: '700px', margin: '0 auto 30px auto' }}>
            Start earning rewards for your recycling efforts while contributing to a healthier planet.
          </Typography>
          {currentUser ? (
            <Button 
              variant="contained" 
              size="large"
              component={Link}
              to="/dashboard"
              style={{ backgroundColor: '#2e7d32', color: 'white', padding: '10px 30px' }}
            >
              Go to Dashboard
            </Button>
          ) : (
            <Button 
              variant="contained" 
              size="large"
              component={Link}
              to="/login"
              style={{ backgroundColor: '#2e7d32', color: 'white', padding: '10px 30px' }}
            >
              Get Started Now
            </Button>
          )}
        </Paper>
      </Container>
    </>
  );
};

export default Home;