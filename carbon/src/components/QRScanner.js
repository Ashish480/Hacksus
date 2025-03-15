import React, { useState, useEffect } from 'react';
import { 
  Container, Typography, Paper, Button, 
  CircularProgress, Box 
} from '@mui/material';
import { doc, updateDoc, getDoc, serverTimestamp } from 'firebase/firestore';
import { Html5Qrcode } from "html5-qrcode";
import { db } from '../firebase';
import { useAuth } from '../contexts/AuthContext';

const QRScanner = () => {
  const { currentUser } = useAuth();
  const [scanning, setScanning] = useState(false);
  const [connected, setConnected] = useState(false);
  const [machineId, setMachineId] = useState(null);
  const [error, setError] = useState('');
  const [countdown, setCountdown] = useState(0);
  const [scannerInitialized, setScannerInitialized] = useState(false);
  const [machine, setMachine] = useState(null);
  
  // Handle QR code scanning
  const handleScan = async (decodedText) => {
    if (!decodedText || connected) return;
    
    try {
      const machineIdTrimmed = decodedText.trim();
      console.log("QR Code scanned:", machineIdTrimmed);
      
      // First just fetch the machine doc to verify it exists
      const machineRef = doc(db, 'machines', machineIdTrimmed);
      const machineDoc = await getDoc(machineRef);
      
      if (!machineDoc.exists()) {
        setError('Invalid QR code. Machine not found.');
        return;
      }
      
      const machineData = machineDoc.data();
      console.log("Machine data:", machineData);
      
      // Check if machine is already in use
      if (machineData.currentSession && machineData.currentSession !== currentUser.uid) {
        setError('Machine is currently in use by another user.');
        return;
      }
      
      // Store machine data
      setMachine(machineData);
      
      // Try to update the machine with proper error handling
      try {
        await updateDoc(machineRef, {
          currentSession: currentUser.uid,
          lastActive: serverTimestamp(),
          status: 'active'
        });
      } catch (updateError) {
        console.error("Permission error on update:", updateError);
        
        // If we can't update, we might still be able to use the machine in read-only mode
        setError('Warning: Unable to update machine status. Limited functionality available.');
        
        // Continue with connection despite error
      }
      
      setMachineId(machineIdTrimmed);
      setConnected(true);
      setCountdown(120);
      setScanning(false);
    } catch (error) {
      console.error("Error connecting to machine:", error);
      
      // More specific error messages
      if (error.code === 'permission-denied') {
        setError('You do not have permission to use this machine.');
      } else {
        setError(`Connection error: ${error.message}`);
      }
    }
  };

  const handleError = (err) => {
    // Only log errors, don't disrupt the scanning process
    console.error("QR Code Error:", err);
  };

  // Start QR code scanning
  const startScan = () => {
    if (!currentUser) {
      setError('You must be logged in to connect to machines.');
      return;
    }
    
    setError('');
    setScanning(true);
    setScannerInitialized(false);
  };

  // Stop QR scanning
  const stopScan = () => {
    setScanning(false);
  };

  // Disconnect from machine
  const disconnect = async () => {
    if (machineId) {
      try {
        const machineRef = doc(db, 'machines', machineId);
        
        // Check if the machine document exists and we have permission to update it
        const machineDoc = await getDoc(machineRef);
        if (machineDoc.exists()) {
          try {
            await updateDoc(machineRef, {
              currentSession: null,
              status: 'idle'
            });
            console.log("Successfully disconnected from machine");
          } catch (updateError) {
            console.error("Unable to update machine status on disconnect:", updateError);
            // Continue with local disconnection even if we can't update the machine
          }
        }
      } catch (error) {
        console.error("Error checking machine on disconnect:", error);
      }
    }
    
    // Always disconnect locally regardless of server-side success
    setConnected(false);
    setMachineId(null);
    setCountdown(0);
    setMachine(null);
  };

  // Initialize scanner
  useEffect(() => {
    let html5QrCode = null;
    
    const initializeScanner = async () => {
      if (!scanning || scannerInitialized) return;
      
      try {
        // First make sure there's a scanner element
        const scannerElement = document.getElementById("qr-video");
        if (!scannerElement) {
          console.error("Scanner element not found");
          setError('Scanner element not found');
          setScanning(false);
          return;
        }
        
        // Check and request camera permission
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          stream.getTracks().forEach(track => track.stop());
        } catch (err) {
          console.error("Camera permission denied:", err);
          setError('Camera access denied. Please check permissions.');
          setScanning(false);
          return;
        }
        
        // Clean up any existing scanner instances
        try {
          const existingScanner = new Html5Qrcode("qr-video");
          await existingScanner.clear();
        } catch (err) {
          // Ignore errors here, just trying to clean up
        }
        
        // Create a new instance of the QR code scanner
        html5QrCode = new Html5Qrcode("qr-video");
        
        html5QrCode.start(
          { facingMode: "environment" },
          {
            fps: 10,
            qrbox: { width: 250, height: 250 }
          },
          handleScan,
          handleError
        ).then(() => {
          console.log("Scanner started successfully");
          setScannerInitialized(true);
        }).catch(err => {
          console.error("Scanner start error:", err);
          setError('Failed to start scanner. Please try again.');
          setScanning(false);
        });
      } catch (error) {
        console.error("Scanner initialization error:", error);
        setError('Camera error. Please reload and try again.');
        setScanning(false);
      }
    };
    
    if (scanning) {
      // Small delay to ensure DOM is ready
      setTimeout(initializeScanner, 500);
    }
    
    // Cleanup function
    return () => {
      if (html5QrCode) {
        html5QrCode.stop().then(() => {
          console.log("Scanner stopped");
        }).catch(err => {
          console.error("Error stopping scanner:", err);
        });
      }
    };
  }, [scanning]);

  // Cleanup scanner when no longer scanning or connected
  useEffect(() => {
    if (!scanning && document.getElementById("qr-video")) {
      try {
        const existingScanner = new Html5Qrcode("qr-video");
        existingScanner.stop().catch(err => {
          console.error("Error stopping existing scanner:", err);
        });
      } catch (err) {
        // Ignore errors here, just trying to clean up
      }
    }
  }, [scanning]);

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      // Try to stop any existing scanner instance
      try {
        if (document.getElementById("qr-video")) {
          const existingScanner = new Html5Qrcode("qr-video");
          existingScanner.stop().catch(console.error);
        }
      } catch (err) {
        console.error("Cleanup error:", err);
      }
      
      // Disconnect if still connected
      if (connected && machineId) {
        const machineRef = doc(db, 'machines', machineId);
        getDoc(machineRef).then(doc => {
          if (doc.exists() && doc.data().currentSession === currentUser?.uid) {
            updateDoc(machineRef, {
              currentSession: null,
              status: 'idle'
            }).catch(console.error);
          }
        }).catch(console.error);
      }
    };
  }, []);

  // Countdown timer for auto-disconnect
  useEffect(() => {
    let interval;
    if (connected && countdown > 0) {
      interval = setInterval(() => {
        setCountdown(prev => prev - 1);
      }, 1000);
    } else if (countdown === 0 && connected) {
      console.log("Session timeout - disconnecting");
      disconnect();
    }
    return () => clearInterval(interval);
  }, [connected, countdown]);

  // Activity checker to refresh session
  useEffect(() => {
    let activityChecker;
    if (connected && machineId && currentUser) {
      activityChecker = setInterval(async () => {
        try {
          const machineRef = doc(db, 'machines', machineId);
          const machineDoc = await getDoc(machineRef);
          
          if (!machineDoc.exists()) {
            console.log("Machine no longer exists");
            setConnected(false);
            setMachineId(null);
            setCountdown(0);
            return;
          }
          
          const machineData = machineDoc.data();
          
          // Check if we're still the current session
          if (machineData.currentSession !== currentUser.uid) {
            console.log("No longer the current session");
            setConnected(false);
            setMachineId(null);
            setCountdown(0);
            return;
          }
          
          // Try to update the lastActive timestamp
          try {
            await updateDoc(machineRef, {
              lastActive: serverTimestamp()
            });
            setCountdown(120);
          } catch (updateError) {
            console.error("Cannot update lastActive:", updateError);
            // Keep the connection anyway
          }
        } catch (error) {
          console.error("Error checking activity:", error);
        }
      }, 30000); // Check every 30 seconds
    }

    return () => {
      if (activityChecker) clearInterval(activityChecker);
    };
  }, [connected, machineId, currentUser]);

  // Format time remaining in mm:ss
  const formatTimeRemaining = () => {
    const minutes = Math.floor(countdown / 60);
    const seconds = countdown % 60;
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
  };

  return (
    <Container maxWidth="sm" style={{ marginTop: '2rem' }}>
      <Paper elevation={3} style={{ padding: '2rem' }}>
        <Typography variant="h4" align="center" gutterBottom>
          Scan QR Code
        </Typography>
        
        {error && (
          <Typography variant="body2" color="error" align="center" gutterBottom>
            {error}
          </Typography>
        )}
        
        {connected ? (
          <Box display="flex" flexDirection="column" alignItems="center">
            <Typography variant="h6">Connected to Machine</Typography>
            <Typography variant="body1" gutterBottom>
              Machine ID: {machineId}
            </Typography>
            {machine && machine.name && (
              <Typography variant="body1" gutterBottom>
                Name: {machine.name}
              </Typography>
            )}
            <Typography variant="body1">
              Time remaining: {formatTimeRemaining()}
            </Typography>
            <Button 
              variant="contained" 
              color="secondary" 
              onClick={disconnect}
              style={{ marginTop: '1rem' }}
            >
              Disconnect
            </Button>
          </Box>
        ) : scanning ? (
          <Box display="flex" flexDirection="column" alignItems="center">
            <div 
              id="qr-video" 
              style={{ 
                width: '300px', 
                height: '300px', 
                margin: '1rem auto', 
                backgroundColor: '#f0f0f0' 
              }} 
            />
            {scannerInitialized ? (
              <Typography variant="body2" color="primary" gutterBottom>
                Point camera at QR code
              </Typography>
            ) : (
              <Box display="flex" alignItems="center" mt={2}>
                <CircularProgress size={24} style={{ marginRight: '10px' }} />
                <Typography variant="body2">Initializing camera...</Typography>
              </Box>
            )}
            <Button 
              variant="contained" 
              color="primary" 
              onClick={stopScan}
              style={{ marginTop: '1rem' }}
            >
              Cancel
            </Button>
          </Box>
        ) : (
          <Box display="flex" flexDirection="column" alignItems="center" marginTop="1rem">
            <Typography variant="body1" gutterBottom>
              Scan a QR code to connect to a machine
            </Typography>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={startScan}
              disabled={!currentUser}
            >
              Start Scanning
            </Button>
            {!currentUser && (
              <Typography variant="body2" color="error" style={{ marginTop: '0.5rem' }}>
                You must be logged in to connect to machines
              </Typography>
            )}
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default QRScanner;