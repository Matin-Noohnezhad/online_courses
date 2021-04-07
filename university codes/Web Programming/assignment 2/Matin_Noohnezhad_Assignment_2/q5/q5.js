var text;
var name = window.sessionStorage.getItem("username");
if (name == "null") {
    text = "<form>\n" +
        "        <br> نام خود را وارد کنید:\n" +
        "        <input id=\"i1\" type=\"text\" name=\"firstname\" required>\n" +
        "        <button onclick=\"func()\">OK</button>\n" +
        "        <br>\n" +
        "    </form>";
    document.getElementById("d1").innerHTML = text;
} else {
    var text = "<p>";
    text += name;
    text += " به این سایت خوش آمدید.";
    text += "</p>";
    document.getElementById("d1").innerHTML = text;
}

function func() {
    var username = document.getElementById("i1").value;
    if (username == "") {
        window.alert("نام کاربری خود را وارد کنید.");
    } else {
        window.sessionStorage.setItem("username", username);
        var text = "<p>";
        text += username;
        text += " به این سایت خوش آمدید.";
        text += "</p>";
        document.getElementById("d1").innerHTML = text;
    }


}