var arr = [50000, 100000, 200000];

function validate1() {
    var bol = mobileNoChecker();
    if (bol) {
        bol = emailChecker();
        if (bol) {
            document.getElementById("first").style.display = "none";
            document.getElementById("second").style.display = "block";
            document.getElementById("third").style.display = "none";
        }
    }
}

function validate2() {
    document.getElementById("first").style.display = "none";
    document.getElementById("second").style.display = "none";
    document.getElementById("third").style.display = "block";
}

function validate3() {
    checkBox = document.getElementById("myCheck");
    if (checkBox.checked == true) {
        return true;
    } else {
        document.getElementById("lastP").style.display = 'block';
        return false
    }
}

function mobileNoChecker() {
    var mobileN = document.forms["myForm"]["telephone"].value;
    var str = mobileN.toString();
    if (str.length == 11) {
        if (str.substring(0, 1) == 0 && str.substring(1, 2) == 9) {
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

function mobileCheckOnType() {
    var mobileN = document.forms["myForm"]["telephone"].value;
    var str = mobileN.toString();
    if (str.substring(0, 1) == '0') {
        if (str.length > 1) {
            if (str.substring(1, 2) == '9') {
                //do nothing
            } else
                document.forms["myForm"]["telephone"].value = '0';
        }
    } else {
        document.forms["myForm"]["telephone"].value = '';
    }

}

function emailChecker() {
    var email = document.forms["myForm"]["email"].value;
    if (!(email.endsWith('@gmail.com') || email.endsWith('@yahoo.com'))) {
        // window.alert("email format is not correct");
        return false;
    }
    return true;
}

function dropdownSelect() {
    var val = document.getElementById('mySelect').value;
    var elements = document.getElementsByClassName('divs');
    var label = document.getElementById("myLabel");
    if (val == 'first') {
        for (var i = 0; i < elements.length; i++) {
            elements[i].style.display = 'none';
        }
        label.innerHTML = "قیمت: " + arr[0].toString() + " تومان";
    } else if (val == 'second') {
        for (var i = 0; i < elements.length; i++) {
            elements[i].style.display = 'none';
        }
        label.innerHTML = "قیمت: " + arr[1].toString() + " تومان";
    } else if (val == 'third') {
        for (var i = 0; i < elements.length; i++) {
            elements[i].style.display = 'inline-block';
        }
        label.innerHTML = "قیمت: " + arr[2].toString() + " تومان";
    }

}