import React, { useState, useEffect } from 'react';
import { 
  Container, Typography, Paper, Grid, Box, 
  CircularProgress, Card, CardContent 
} from '@mui/material';
import { 
  collection, query, where, orderBy, limit, 
  getDocs, doc, getDoc 
} from 'firebase/firestore';
import { db } from '../firebase';
import { useAuth } from '../contexts/AuthContext';
import { Doughnut } from 'react-chartjs-2';

const Dashboard = () => {
  const { currentUser } = useAuth();
  const [userPoints, setUserPoints] = useState(0);
  const [transactions, setTransactions] = useState([]);
  const [wasteStats, setWasteStats] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        if (!currentUser) return;
        
        // Get user data
        const userRef = doc(db, 'users', currentUser.uid);
        const userDoc = await getDoc(userRef);
        
        if (userDoc.exists()) {
          setUserPoints(userDoc.data().points || 0);
        }
        
        // Get recent transactions
        const transactionsRef = collection(db, 'transactions');
        const q = query(
          transactionsRef,
          where('userId', '==', currentUser.uid),
          orderBy('timestamp', 'desc'),
          limit(10)
        );
        
        const querySnapshot = await getDocs(q);
        const transactionsList = [];
        const wasteTypeCount = {};
        
        querySnapshot.forEach((doc) => {
          const data = doc.data();
          transactionsList.push({
            id: doc.id,
            ...data,
            timestamp: data.timestamp?.toDate()
          });
          
          // Count waste types
          const wasteType = data.wasteType;
          wasteTypeCount[wasteType] = (wasteTypeCount[wasteType] || 0) + 1;
        });
        
        setTransactions(transactionsList);
        setWasteStats(wasteTypeCount);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching user data:", error);
        setLoading(false);
      }
    };
    
    fetchUserData();
  }, [currentUser]);

  // Prepare chart data
  const chartData = {
    labels: Object.keys(wasteStats),
    datasets: [
      {
        data: Object.values(wasteStats),
        backgroundColor: [
          '#4CAF50', // Green for paper
          '#2196F3', // Blue for plastic
          '#FFC107', // Yellow for metal
          '#9C27B0', // Purple for glass
          '#FF5722', // Orange for organic
        ],
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
      },
    },
  };

  if (loading) {
    return (
      <Container style={{ display: 'flex', justifyContent: 'center', marginTop: '50px' }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="md" style={{ marginTop: '20px' }}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper elevation={3} style={{ padding: '20px', backgroundColor: '#f5f9f5' }}>
            <Typography variant="h4" style={{ color: '#2e7d32' }}>
              Welcome to Carbon Bytes!
            </Typography>
            <Typography variant="body1" style={{ marginTop: '10px' }}>
              Thank you for contributing to a cleaner environment. Your small actions make a big difference!
            </Typography>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Your Points
              </Typography>
              <Typography variant="h3" style={{ color: '#2e7d32' }}>
                {userPoints}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Keep recycling to earn more points!
              </Typography>
              <Box mt={2}>
                <Typography variant="body2">
                  Points can be redeemed for rewards (Coming Soon)
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Waste Breakdown
              </Typography>
              <Box height={200}>
                {Object.keys(wasteStats).length > 0 ? (
                  <Doughnut data={chartData} options={chartOptions} />
                ) : (
                  <Box display="flex" justifyContent="center" alignItems="center" height="100%">
                    <Typography variant="body1" color="textSecondary">
                      No waste data available yet
                    </Typography>
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Recent Activity
              </Typography>
              {transactions.length > 0 ? (
                transactions.map((transaction) => (
                  <Paper 
                    key={transaction.id} 
                    elevation={1} 
                    style={{ padding: '10px', marginBottom: '10px' }}
                  >
                    <Grid container justifyContent="space-between" alignItems="center">
                      <Grid item>
                        <Typography variant="body1">
                          {transaction.wasteType.charAt(0).toUpperCase() + transaction.wasteType.slice(1)}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          {transaction.timestamp?.toLocaleDateString()} {transaction.timestamp?.toLocaleTimeString()}
                        </Typography>
                      </Grid>
                      <Grid item>
                        <Typography variant="body1" style={{ color: '#2e7d32', fontWeight: 'bold' }}>
                          +{transaction.points} points
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
      </Grid>
    </Container>
  );
};

export default Dashboard;