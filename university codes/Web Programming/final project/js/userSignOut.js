function sign_out(bol) {

    if (bol) {
        destroy_session();
        window.open("../index.php", "_self");
    } else {
        window.open("../index.php", "_self");
    }
}

function destroy_session() {
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.open("GET", "destroy_session.php", true);
    xmlhttp.send();
}