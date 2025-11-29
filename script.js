document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const landingPage = document.getElementById('landing-page');
    const analysisPage = document.getElementById('analysis-page');
    const authModal = document.getElementById('auth-modal');
    const loginForm = document.getElementById('login-form');
    const signupForm = document.getElementById('signup-form');
    const tabLogin = document.getElementById('tab-login');
    const tabSignup = document.getElementById('tab-signup');
    const authMessage = document.getElementById('auth-message');
    
    // Check Session on Load
    checkSession();

    async function checkSession() {
        try {
            const res = await fetch('/check_auth');
            const data = await res.json();
            if (data.is_logged_in) {
                showPage(analysisPage);
                document.getElementById('user-profile-btn').innerHTML = `<i class="fas fa-user"></i> ${data.name}`;
            } else {
                showPage(landingPage);
            }
        } catch (e) { showPage(landingPage); }
    }

    // Toggle Pages
    function showPage(page) {
        document.querySelectorAll('.page-view').forEach(p => p.classList.remove('active'));
        page.classList.add('active');
    }

    // Modal & Tabs Logic
    document.getElementById('get-started-btn').onclick = () => authModal.classList.add('visible');
    document.getElementById('show-login-nav').onclick = () => authModal.classList.add('visible');
    document.getElementById('close-modal-btn').onclick = () => authModal.classList.remove('visible');

    tabLogin.onclick = () => {
        loginForm.style.display = 'block';
        signupForm.style.display = 'none';
        tabLogin.style.borderBottom = '2px solid var(--color-primary)';
        tabLogin.style.color = 'var(--color-primary)';
        tabSignup.style.borderBottom = 'none';
        tabSignup.style.color = '#999';
    };

    tabSignup.onclick = () => {
        loginForm.style.display = 'none';
        signupForm.style.display = 'block';
        tabSignup.style.borderBottom = '2px solid var(--color-primary)';
        tabSignup.style.color = 'var(--color-primary)';
        tabLogin.style.borderBottom = 'none';
        tabLogin.style.color = '#999';
    };

    // Handle Login
    loginForm.onsubmit = async (e) => {
        e.preventDefault();
        const formData = new FormData(loginForm);
        const data = Object.fromEntries(formData.entries());
        
        const res = await fetch('/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        const result = await res.json();
        
        if (result.status === 'success') {
            authModal.classList.remove('visible');
            checkSession(); // Refresh to update UI
        } else {
            authMessage.style.color = 'red';
            authMessage.textContent = result.message;
        }
    };

    // Handle Signup
    signupForm.onsubmit = async (e) => {
        e.preventDefault();
        const formData = new FormData(signupForm);
        const data = Object.fromEntries(formData.entries());
        
        const res = await fetch('/signup', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        const result = await res.json();
        
        if (result.status === 'success') {
            authModal.classList.remove('visible');
            checkSession();
        } else {
            authMessage.style.color = 'red';
            authMessage.textContent = result.message;
        }
    };

    // Handle Logout
    window.logout = async () => {
        await fetch('/logout');
        location.reload();
    };

    // Keep existing Analysis Logic (Simplified)
    document.getElementById('uploadForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = document.getElementById('analyzeButton');
        const fileInput = document.getElementById('videoFile');
        const status = document.getElementById('statusMessage');
        const results = document.getElementById('resultsContainer');

        if (!fileInput.files.length) return;
        
        const formData = new FormData();
        formData.append('video', fileInput.files[0]);
        
        btn.disabled = true;
        status.textContent = "Analyzing...";
        status.className = "status-message";

        try {
            const res = await fetch('/predict_video', { method: 'POST', body: formData });
            const data = await res.json();
            
            if (data.status === 'success') {
                status.textContent = "Done!";
                status.classList.add('success');
                
                let html = '<table><tr><th>Time</th><th>Face</th><th>Voice</th><th>Fused</th><th>Conf</th></tr>';
                data.analysis.forEach(row => {
                    html += `<tr><td>${row.time}s</td><td>${row.face}</td><td>${row.voice}</td><td><strong>${row.fused}</strong></td><td>${row.conf}</td></tr>`;
                });
                results.innerHTML = html + '</table>';
            } else {
                status.textContent = data.error || "Failed";
                status.classList.add('error');
            }
        } catch (err) {
            status.textContent = "Server Error";
        } finally {
            btn.disabled = false;
        }
    });
});