// Initialize Supabase client
const supabaseUrl = 'YOUR_SUPABASE_URL'
const supabaseKey = 'YOUR_SUPABASE_ANON_KEY'
const supabase = supabase.createClient(supabaseUrl, supabaseKey)

// Initialize Firebase
const auth = firebase.auth();

// Initialize auth state
function initAuth() {
    // Check if user is already signed in
    auth.onAuthStateChanged((user) => {
        if (user) {
            // User is signed in
            openTab(null, 'chat');
        } else {
            // User is signed out
            openTab(null, 'signin');
        }
    });

    // Set up sign in form
    const signinForm = document.getElementById('signin-form');
    if (signinForm) {
        signinForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('signin-email').value;
            const password = document.getElementById('signin-password').value;
            
            try {
                const userCredential = await auth.signInWithEmailAndPassword(email, password);
                openTab(null, 'chat');
            } catch (error) {
                document.getElementById('signin-error').textContent = error.message;
            }
        });
    }

    // Set up sign up form
    const signupForm = document.getElementById('signup-form');
    if (signupForm) {
        signupForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('signup-username').value;
            const email = document.getElementById('signup-email').value;
            const password = document.getElementById('signup-password').value;
            const confirmPassword = document.getElementById('signup-confirm-password').value;

            if (password !== confirmPassword) {
                document.getElementById('signup-error').textContent = 'Passwords do not match';
                return;
            }

            try {
                const userCredential = await auth.createUserWithEmailAndPassword(email, password);
                // Update profile with username
                await userCredential.user.updateProfile({
                    displayName: username
                });
                openTab(null, 'chat');
            } catch (error) {
                document.getElementById('signup-error').textContent = error.message;
            }
        });
    }
}

// Sign out function
function signOut() {
    auth.signOut().then(() => {
        openTab(null, 'signin');
    }).catch((error) => {
        console.error('Error signing out:', error);
    });
} 