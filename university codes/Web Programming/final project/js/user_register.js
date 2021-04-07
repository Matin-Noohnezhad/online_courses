function formValidation() {

    var email = document.forms["myForm"]["email"].value;
    var username = document.forms["myForm"]["username"].value;
    var pass = document.forms["myForm"]["pass"].value;
    var repass = document.forms["myForm"]["repass"].value;
    var valid = true;
    //check if any field is empty
    valid = checkNotEmpty(email, username, pass, repass, valid);
    //check if email format is correct
    valid = checkEmailFormat(email, valid);
    //check if password and re-password is equal
    valid = passRepassEquality(pass, repass, valid);

    return valid;
}

function checkNotEmpty(email, username, pass, repass, valid) {

    if (email.length == 0) {
        valid = false;
        document.getElementById("err1").innerHTML = "پر کردن این فیلد اجباریست.*";
    } else {
        document.getElementById("err1").innerHTML = "";
    }
    if (username.length == 0) {
        valid = false;
        document.getElementById("err2").innerHTML = "پر کردن این فیلد اجباریست.*";
    } else {
        document.getElementById("err2").innerHTML = "";
    }

    if (pass.length == 0) {
        valid = false;
        document.getElementById("err3").innerHTML = "پر کردن این فیلد اجباریست.*";
    } else {
        document.getElementById("err3").innerHTML = "";
    }

    if (repass.length == 0) {
        valid = false;
        document.getElementById("err4").innerHTML = "پر کردن این فیلد اجباریست.*";
    } else {
        document.getElementById("err4").innerHTML = "";
    }
    return valid;
}

function checkEmailFormat(email, valid) {
    if (valid) {
        if (!(email.endsWith('@gmail.com') || email.endsWith('@yahoo.com') || email.endsWith('@bihe.org'))) {
            document.getElementById("err1").innerHTML = "فرمت ایمیل وارد شده معتبر نیست.";
            valid = false;
        }
    }
    return valid;
}

function passRepassEquality(pass, repass, valid) {
    if (valid) {
        if (pass != repass) {
            document.getElementById("err4").innerHTML = "با رمز عبور وارد شده یکی نیست.";
            valid = false;
        }
    }
    return valid;
}