function addCategory() {
    var text = document.getElementById("newCategory").value;
    var bol = true;
    if (text.length == 0) {
        document.getElementById("error").innerHTML = 'نام دسته را بنویسید.*';
        bol = false;
    } else {
        document.getElementById("error").innerHTML = '';

    }
    return bol;
}

