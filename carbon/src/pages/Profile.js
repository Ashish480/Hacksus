// src/pages/Profile.js
import React, { useState, useEffect } from 'react';
import { 
  Container, Typography, Paper, Grid, Avatar,
  Button, Card, CardContent, Divider, Box,
  CircularProgress, List, ListItem, ListItemText,
  ListItemIcon
} from '@mui/material';
import { 
  doc, getDoc, collection, query, where, 
  orderBy, limit, getDocs 
} from 'firebase/firestore';
import { db } from '../firebase';
import { useAuth } from '../contexts/AuthContext';
import { Bar } from 'react-chartjs-2';

const Profile = () => {
  const { currentUser } = useAuth();
  const [userData, setUserData] = useState(null);
  const [stats, setStats] = useState({
    totalWaste: 0,
    totalPoints: 0,
    wasteByType: {},
    recentActivity: []
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchUserProfile = async () => {
      try {
        if (!currentUser) return;
        
        // Get user data
        const userRef = doc(db, 'users', currentUser.uid);
        const userDoc = await getDoc(userRef);
        
        if (userDoc.exists()) {
          setUserData(userDoc.data());
        }
        
        // Get user transactions
        const transactionsRef = collection(db, 'transactions');
        const q = query(
          transactionsRef,
          where('userId', '==', currentUser.uid),
          orderBy('timestamp', 'desc')
        );
        
        const querySnapshot = await getDocs(q);
        
        let totalItems = 0;
        let totalPointsEarned = 0;
        const wasteTypes = {};
        const recentActivity = [];
        
        querySnapshot.forEach((doc) => {
          const data = doc.data();
          totalItems++;
          totalPointsEarned += data.points || 0;
          
          // Count waste types
          const wasteType = data.wasteType;
          wasteTypes[wasteType] = (wasteTypes[wasteType] || 0) + 1;
          
          // Add to recent activity if within limit
          if (recentActivity.length < 5) {
            recentActivity.push({
              id: doc.id,
              ...data,
              timestamp: data.timestamp?.toDate()
            });
          }
        });
        
        setStats({
          totalWaste: totalItems,
          totalPoints: totalPointsEarned,
          wasteByType: wasteTypes,
          recentActivity
        });
        
        setLoading(false);
      } catch (error) {
        console.error("Error fetching user profile:", error);
        setLoading(false);
      }
    };
    
    fetchUserProfile();
  }, [currentUser]);

  // Prepare chart data
  const chartData = {
    labels: Object.keys(stats.wasteByType).map(type => 
      type.charAt(0).toUpperCase() + type.slice(1)
    ),
    datasets: [
      {
        label: 'Items Recycled',
        data: Object.values(stats.wasteByType),
        backgroundColor: [
          '#4CAF50', // Green for paper
          '#2196F3', // Blue for plastic
          '#FFC107', // Yellow for metal
          '#9C27B0', // Purple for glass
          '#FF5722', // Orange for organic
        ],
        borderColor: [
          '#388E3C',
          '#1976D2',
          '#FFA000',
          '#7B1FA2',
          '#E64A19',
        ],
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          precision: 0
        }
      }
    }
  };

  if (loading) {
    return (
      <Container style={{ display: 'flex', justifyContent: 'center', marginTop: '50px' }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="md" style={{ marginTop: '20px', marginBottom: '40px' }}>
      <Paper elevation={3} style={{ padding: '30px', backgroundColor: '#f9f9f9' }}>
        <Grid container spacing={4}>
          {/* User Profile Section */}
          <Grid item xs={12}>
            <Box display="flex" alignItems="center">
              <Avatar 
                src={userData?.photoURL} 
                alt={userData?.displayName}
                style={{ width: '80px', height: '80px', marginRight: '20px' }}
              />
              <div>
                <Typography variant="h4" style={{ color: '#2e7d32' }}>
                  {userData?.displayName}
                </Typography>
                <Typography variant="body1" color="textSecondary">
                  {userData?.email}
                </Typography>
                <Typography variant="body2" style={{ marginTop: '5px' }}>
                  Member since: {userData?.createdAt?.toDate().toLocaleDateString()}
                </Typography>
              </div>
            </Box>
          </Grid>
          
          <Grid item xs={12}>
            <Divider />
          </Grid>
          
          {/* Stats Section */}
          <Grid item xs={12} md={4}>
            <Card style={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Environmental Impact
                </Typography>
                
                <List>
                  <ListItem>
                    <ListItemText 
                      primary="Total Items Recycled" 
                      secondary={stats.totalWaste}
                      primaryTypographyProps={{ variant: 'body2' }}
                      secondaryTypographyProps={{ variant: 'h6', style: { color: '#2e7d32' } }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemText 
                      primary="Total Points Earned" 
                      secondary={stats.totalPoints}
                      primaryTypographyProps={{ variant: 'body2' }}
                      secondaryTypographyProps={{ variant: 'h6', style: { color: '#2e7d32' } }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemText 
                      primary="Carbon Saved" 
                      secondary={`${(stats.totalWaste * 0.5).toFixed(1)} kg`}
                      primaryTypographyProps={{ variant: 'body2' }}
                      secondaryTypographyProps={{ variant: 'h6', style: { color: '#2e7d32' } }}
                    />
                  </ListItem>
                </List>
                
                <Box mt={2} textAlign="center">
                  <Button 
                    variant="outlined" 
                    color="primary" 
                    disabled 
                    style={{ marginTop: '10px' }}
                  >
                    Redeem Points (Coming Soon)
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          {/* Chart Section */}
          <Grid item xs={12} md={8}>
            <Card style={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Recycling Breakdown
                </Typography>
                <Box height={300}>
                  {Object.keys(stats.wasteByType).length > 0 ? (
                    <Bar data={chartData} options={chartOptions} />
                  ) : (
                    <Box display="flex" justifyContent="center" alignItems="center" height="100%">
                      <Typography variant="body1" color="textSecondary">
                        No recycling data available yet
                      </Typography>
                    </Box>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          {/* Recent Activity Section */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Recent Activity
                </Typography>
                
                {stats.recentActivity.length > 0 ? (
                  stats.recentActivity.map((activity) => (
                    <Paper 
                      key={activity.id} 
                      elevation={1} 
                      style={{ padding: '15px', marginBottom: '10px' }}
                    >
                      <Grid container justifyContent="space-between" alignItems="center">
                        <Grid item>
                          <Typography variant="body1">
                            Recycled {activity.wasteType.charAt(0).toUpperCase() + activity.wasteType.slice(1)}
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            {activity.timestamp?.toLocaleDateString()} at {activity.timestamp?.toLocaleTimeString()}
                          </Typography>
                        </Grid>
                        <Grid item>
                          <Typography variant="body1" style={{ color: '#2e7d32', fontWeight: 'bold' }}>
                            +{activity.points} points
                          </Typography>
                        </Grid>
                      </Grid>
                    </Paper>
                  ))
                ) : (
                  <Typography variant="body1" color="textSecondary">
                    No recent activity
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
          
          {/* Environmental Impact Section */}
          <Grid item xs={12}>
            <Card style={{ backgroundColor: '#e8f5e9' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Your Environmental Impact
                </Typography>
                
                <Grid container spacing={3}>
                  <Grid item xs={12} md={4}>
                    <Box textAlign="center" p={2}>
                      <Typography variant="h5" style={{ color: '#2e7d32' }}>
                        {(stats.totalWaste * 0.5).toFixed(1)} kg
                      </Typography>
                      <Typography variant="body2">
                        CO₂ Emissions Saved
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Box textAlign="center" p={2}>
                      <Typography variant="h5" style={{ color: '#2e7d32' }}>
                        {(stats.totalWaste * 0.1).toFixed(1)} L
                      </Typography>
                      <Typography variant="body2">
                        Water Saved
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Box textAlign="center" p={2}>
                      <Typography variant="h5" style={{ color: '#2e7d32' }}>
                        {Math.floor(stats.totalWaste / 10)}
                      </Typography>
                      <Typography variant="body2">
                        Trees Saved
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>
    </Container>
  );
};

export default Profile;