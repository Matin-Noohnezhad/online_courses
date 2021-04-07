function go_to_register() {
    window.open("./php/register.php", "_self");
}

function go_to_basket() {
    window.open("./php/basket.php", "_self");
}

function search() {
    var text = document.getElementById("search_text").value;
    document.getElementById("content_div").innerHTML = '<p>' + text + '</p>';
}

function sign_out() {
    window.open("./php/user_sign_out.php", "_self");
}


function myPostForCategory(categoryName) {
// <form method="post" action="./php/goodByCategory.php" id="goodByCategoryForm" name="goodByCategoryForm">
//  must write something here...
//         </form>
    sendingInput = '<input name="categoryName" value="' + categoryName + '" style="display: none">'
    var form = document.getElementById("goodByCategoryForm");
    form.innerHTML = sendingInput;
    form.submit();

}

function myPostForStore(storeName) {
// <form method="post" action="./php/goodByCategory.php" id="goodByStoreForm" name="goodByStoreForm">
//  must write something here...
//         </form>
    sendingInput = '<input name="storeName" value="' + storeName + '" style="display: none">'
    var form = document.getElementById("goodByStoreForm");
    form.innerHTML = sendingInput;
    form.submit();

}

function formPost(goodId) {
    // window.alert(goodId);

    sendingInput = '<input name="good_id" value=' + goodId + ' style="display: none">';
    var form = document.getElementById("5Goods");
    form.innerHTML = sendingInput;
    form.submit();

}
