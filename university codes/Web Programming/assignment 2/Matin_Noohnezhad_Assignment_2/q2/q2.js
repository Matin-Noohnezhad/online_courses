var interests = ["بسکتبال", "کامپیوتر", "بازی"];
optFill(interests, "select1");

function optFill(interests, id) {
    var answer = '';
    for (x of interests) {
        answer += '<option value=';
        answer += x;
        answer += '">';
        answer += x;
        answer += '</option>';
    }
    document.getElementById(id).innerHTML = answer;
// <option value="volvo">Volvo</option>
//         <option value="saab">Saab</option>
//         <option value="fiat">Fiat</option>
//         <option value="audi">Audi</option>
}

function formValidation() {
    var psw = document.forms["myForm"]["psw"].value;
    var pswc = document.forms["myForm"]["pswc"].value;
    var bol = true;
    if (psw != pswc) {
        window.alert("passwords is not equal");
        bol = false;
    } else {
        var email = document.forms["myForm"]["email"].value;
        if (!(email.endsWith('@gmail.com') || email.endsWith('@yahoo.com'))) {
            window.alert("email format is not correct");
            bol = false;
        }
    }
    return bol;
}

function errorPrint() {
    if (document.forms["myForm"]["psw"].validity.patternMismatch) {
        // window.alert("pattern mismatch ");
        document.getElementById("p1").innerHTML = "فرمت وارد شده برای رمز عبور اشتباه است. باید دارای حرف بزرگ، حرف کوچک، عدد و حداقل 6 حرف باشد.";
        document.getElementById("pass").style.border = "5px solid purple";
    }
    var email = document.forms["myForm"]["email"].value;
    if (!(email.endsWith('@gmail.com') || email.endsWith('@yahoo.com'))) {
        // window.alert("email format is not correct");
        document.getElementById("p2").innerHTML = "فرمت ایمیل وارد شده معتبر نیست.";
        document.getElementById("email").style.border = "5px solid purple";
    }
}