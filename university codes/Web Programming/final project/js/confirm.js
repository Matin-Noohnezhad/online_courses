function confirm(id) {
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function () {
    };
    xmlhttp.open("GET", "confirm_record.php?q=" + id + "&w=" + 1, true);
    xmlhttp.send();
    window.open("./confirm.php", "_self");
}

function noConfirm(id) {
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function () {
    };
    xmlhttp.open("GET", "confirm_record.php?q=" + id + "&w=" + 2, true);
    xmlhttp.send();
    window.open("./confirm.php", "_self");
}
