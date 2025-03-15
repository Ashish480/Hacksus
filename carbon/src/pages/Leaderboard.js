import React, { useState, useEffect } from 'react';
import { 
  Container, Typography, Paper, Table, TableBody, 
  TableCell, TableContainer, TableHead, TableRow, 
  CircularProgress, Box, Avatar 
} from '@mui/material';
import { collection, query, orderBy, limit, onSnapshot } from 'firebase/firestore';
import { db } from '../firebase';
import { useAuth } from '../contexts/AuthContext';

const Leaderboard = () => {
  const { currentUser } = useAuth();
  const [leaderboard, setLeaderboard] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const usersRef = collection(db, 'users');
    const q = query(
      usersRef,
      orderBy('points', 'desc'),
      limit(50)
    );
    
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const users = [];
      let rank = 1;
      
      snapshot.forEach((doc) => {
        users.push({ 
          id: doc.id, 
          rank: rank++,
          ...doc.data() 
        });
      });
      
      setLeaderboard(users);
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="70vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="md" style={{ marginTop: '2rem', marginBottom: '2rem' }}>
      <Typography variant="h4" gutterBottom>Leaderboard</Typography>
      <Typography variant="body1" paragraph>
        Top environmentally conscious users in the Carbon Bytes community!
      </Typography>
      
      <Paper elevation={3}>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow style={{ backgroundColor: '#e8f5e9' }}>
                <TableCell align="center">Rank</TableCell>
                <TableCell>User</TableCell>
                <TableCell align="right">Points</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {leaderboard.map((user) => (
                <TableRow 
                  key={user.id} 
                  style={{ 
                    backgroundColor: user.id === currentUser?.uid ? '#e8f5e9' : 'inherit' 
                  }}
                >
                  <TableCell align="center">
                    {user.rank <= 3 ? (
                      <span style={{ 
                        fontSize: '1.2rem', 
                        color: ['#FFD700', '#C0C0C0', '#CD7F32'][user.rank - 1]
                      }}>
                        {user.rank}
                      </span>
                    ) : user.rank}
                  </TableCell>
                  <TableCell>
                    <Box display="flex" alignItems="center">
                      <Avatar 
                        src={user.photoURL} 
                        alt={user.displayName} 
                        style={{ marginRight: '0.5rem' }}
                      />
                      <Typography>
                        {user.displayName}
                        {user.id === currentUser?.uid && ' (You)'}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell align="right">
                    <Typography variant="h6" style={{ color: '#4caf50' }}>
                      {user.points}
                    </Typography>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    </Container>
  );
};

export default Leaderboard;