import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyAVmc1LpVqETghg8V0ksWbeWc27fGKx71A",
  authDomain: "carbon-bytes-32bde.firebaseapp.com",
  projectId: "carbon-bytes-32bde",
  storageBucket: "carbon-bytes-32bde.firebasestorage.app",
  messagingSenderId: "783067000643",
  appId: "1:783067000643:web:5df789f7b16a3c552b09f6"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
const googleProvider = new GoogleAuthProvider();

export { auth, db, googleProvider };

