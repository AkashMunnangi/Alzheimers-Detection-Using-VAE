const authWrapper = document.querySelector('.auth-wrapper');
document.querySelector('.register-trigger')?.addEventListener('click', e => {
    e.preventDefault();
    authWrapper.classList.add('toggled');
});
document.querySelector('.login-trigger')?.addEventListener('click', e => {
    e.preventDefault();
    authWrapper.classList.remove('toggled');
});
