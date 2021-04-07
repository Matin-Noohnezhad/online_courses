function validation() {

    var email = document.forms["myForm"]["email"].value;
    var pass = document.forms["myForm"]["pass"].value;
    var valid = true;
    //check if any field is empty
    valid = checkNotEmpty(email, pass, valid);
    //check if email format is correct
    valid = checkEmailFormat(email, valid);

    return valid;
}

function checkNotEmpty(email, pass, valid) {

    if (email.length == 0) {
        valid = false;
        document.getElementById("err1").innerHTML = "پر کردن این فیلد اجباریست.*";
    } else {
        document.getElementById("err1").innerHTML = "";
    }

    if (pass.length == 0) {
        valid = false;
        document.getElementById("err2").innerHTML = "پر کردن این فیلد اجباریست.*";
    } else {
        document.getElementById("err2").innerHTML = "";
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