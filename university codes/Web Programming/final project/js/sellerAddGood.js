var fileSelected = false;

function fileSelect() {
    fileSelected = true;
}

function formValidation() {

    var name = document.forms["myForm"]["name"].value;
    var price = document.forms["myForm"]["price"].value;
    var valid = true;
    //
    valid = checkNotEmpty(name, price, fileSelected, valid);


    return valid;
}

function checkNotEmpty(name, price, fileSelected, valid) {
    if (name.length == 0) {
        valid = false;
        document.getElementById("err1").innerHTML = "پر کردن این فیلد اجباریست.*";
    } else {
        document.getElementById("err1").innerHTML = "";
    }
    if (price.length == 0) {
        valid = false;
        document.getElementById("err2").innerHTML = "پر کردن این فیلد اجباریست.*";
    } else {
        document.getElementById("err2").innerHTML = "";
    }
    if (!fileSelected) {
        valid = false;
        document.getElementById("err3").innerHTML = "انتخاب عکس برای محصول اجباریست.*";
    } else {
        document.getElementById("err3").innerHTML = "";
    }


    return valid;
}