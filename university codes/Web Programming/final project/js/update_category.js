function formValidation() {

    var categoryName = document.forms["myForm"]["categoryNames"].value;
    var newName = document.forms["myForm"]["newName"].value;
    if (newName.length == 0) {
        document.getElementById("err1").innerHTML = 'این فیلد باید پر شود.*';
        return false;
    }else if(categoryName == newName){
        document.getElementById("err1").innerHTML = 'این نام با نام قبلی این دسته یکی است.*';
        return false;
    }
    document.getElementById("err1").innerHTML = '';
    return true;

}

