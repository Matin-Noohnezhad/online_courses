document.getElementById("b1").innerHTML = Math.random();


function func() {
    setTimeout(function () {
        var a = Math.random();
        document.getElementById("b1").innerHTML = a;
    }, 3000);

}